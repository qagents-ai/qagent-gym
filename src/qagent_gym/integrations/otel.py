"""Explicit OpenTelemetry wrappers; no global provider or exporter is configured."""

from __future__ import annotations

import asyncio
from typing import Any

from qagent_gym.records import Context, Generation, Reward, Task


def _attributes(task: Task, context: Context | None = None) -> dict:
    attributes = {"qagent_gym.task_id": task.task_id, "qagent_gym.split": task.split}
    if context is not None:
        attributes.update(
            {
                "qagent_gym.namespace": context.namespace,
                "qagent_gym.context_revision": context.revision,
                "qagent_gym.playbook_bullets": len(context.bullets),
                "qagent_gym.memories": len(context.memories),
            }
        )
    return attributes


def _failure(span: Any, exc: BaseException) -> None:
    from opentelemetry.trace import StatusCode

    span.set_attribute(
        "qagent_gym.status", "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
    )
    # Exception messages/stacks may contain input, credentials or provider payloads.
    # Record only a class name in these wrapper-owned spans.
    span.set_attribute("error.type", type(exc).__name__)
    span.set_status(StatusCode.ERROR, "operation failed")


class TracingGenerator:
    """Decorate a Gym generator with metadata-only spans using an injected tracer.

    IDs/namespaces are caller-supplied metadata, so choose non-sensitive identifiers.
    These controls apply to this wrapper, not to other instrumentation in a delegate.
    """

    def __init__(self, generator: Any, tracer: Any) -> None:
        self.generator = generator
        self.tracer = tracer

    async def generate(self, task: Task, context: Context) -> Generation:
        from opentelemetry.trace import StatusCode

        with self.tracer.start_as_current_span(
            "qagent_gym.generate",
            attributes=_attributes(task, context),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await self.generator.generate(task, context)
                if not isinstance(result, Generation):
                    raise TypeError("generator must return Generation")
                span.set_attribute("qagent_gym.status", result.status)
                span.set_attribute("qagent_gym.trace_steps", len(result.steps))
                span.set_status(StatusCode.OK if result.status == "completed" else StatusCode.ERROR)
                return result
            except BaseException as exc:
                _failure(span, exc)
                raise


class TracingEvaluator:
    """Trace evaluator identity and numeric reward without its feedback text."""

    def __init__(self, evaluator: Any, tracer: Any) -> None:
        self.evaluator = evaluator
        self.tracer = tracer

    async def evaluate(self, task: Task, generation: Generation) -> Reward:
        from opentelemetry.trace import StatusCode

        with self.tracer.start_as_current_span(
            "qagent_gym.evaluate",
            attributes=_attributes(task),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                reward = await self.evaluator.evaluate(task, generation)
                if not isinstance(reward, Reward):
                    raise TypeError("evaluator must return Reward")
                span.set_attribute("qagent_gym.reward", reward.value)
                span.set_attribute("qagent_gym.evaluator", reward.evaluator)
                span.set_status(StatusCode.OK)
                return reward
            except BaseException as exc:
                _failure(span, exc)
                raise
