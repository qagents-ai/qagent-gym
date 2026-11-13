"""Adapters using only qagents' public Agent API; importing core Gym needs no provider SDK."""

from __future__ import annotations

import json
import math

from qagents import Agent, Limits

from .prompts import GENERATOR_PROMPT, REFLECTOR_PROMPT
from .records import Context, Episode, Generation, Lesson, Step, Task


class QAgentsGenerator:
    """Run a text-output Agent with a fresh session and finite execution budgets.

    Preserve actual model identity from response metadata when available. The
    journal captures observable text/tool I/O, not hidden reasoning or token logprobs.
    Paused runs are recorded as incomplete; resuming approvals is outside this adapter.
    """

    def __init__(
        self, agent: Agent, *, limits: Limits | None = None, max_context_chars: int = 20000
    ) -> None:
        self.agent = agent
        self.limits = limits or Limits(
            max_turns=8, timeout_s=60, max_total_tokens=16000, max_cost_usd=1.0
        )
        if not self.limits.max_turns or self.limits.max_turns < 1:
            raise ValueError("a finite positive max_turns is required")
        if not self.limits.timeout_s or not math.isfinite(self.limits.timeout_s):
            raise ValueError("a finite positive timeout_s is required")
        if self.limits.timeout_s <= 0 or max_context_chars < 1:
            raise ValueError("timeout_s and max_context_chars must be positive")
        self.max_context_chars = max_context_chars

    async def generate(self, task: Task, context: Context) -> Generation:
        rendered = context.render()
        if len(rendered) > self.max_context_chars:
            raise ValueError("context exceeds max_context_chars; curate or narrow the namespace")
        result = await self.agent.run(
            GENERATOR_PROMPT.format(context=rendered, task=task.prompt),
            limits=self.limits,
        )
        output = ""
        if result.status == "completed":
            if not isinstance(result.output, str):
                raise ValueError("QAgentsGenerator requires a text-output Agent")
            output = result.output
        steps = []
        model = "unknown"
        for message in result.new_messages:
            if message.text:
                steps.append(Step("message", message.text, role=message.role))
            for call in message.tool_calls:
                steps.append(
                    Step(
                        "tool_call",
                        json.dumps(call.arguments, sort_keys=True),
                        role=message.role,
                        name=call.name,
                    )
                )
            for tool_result in message.tool_results:
                steps.append(
                    Step(
                        "tool_result", tool_result.content, role=message.role, name=tool_result.name
                    )
                )
            if message.role == "assistant" and message.metadata.get("model"):
                model = f"{message.metadata.get('provider', 'unknown')}:{message.metadata['model']}"
        return Generation(
            output,
            status=result.status,
            steps=tuple(steps),
            model=model,
            stop_reason=result.stop_reason or "",
        )


class QAgentsReflector:
    """Ask a separately configured qagents agent for validated JSON lessons."""

    def __init__(self, generator: QAgentsGenerator) -> None:
        self.generator = generator

    async def reflect(self, episode: Episode) -> tuple[Lesson, ...]:
        if episode.task.split != "train":
            raise ValueError("held-out episodes cannot be reflected into the playbook")
        result = await self.generator.generate(
            Task(f"reflect:{episode.run_id}", REFLECTOR_PROMPT.format(episode=episode.to_json())),
            episode.context,
        )
        if result.status != "completed":
            raise ValueError(f"reflection did not complete: {result.status}")
        try:
            values = json.loads(result.output)
            if not isinstance(values, list):
                raise ValueError("expected a JSON array")
            return tuple(Lesson(**value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid reflection: {exc}") from exc
