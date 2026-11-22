"""Resume a frozen, offline evaluation after a worker stops and SQLite reopens.

Run: python -m examples.background_evaluation --db evaluation.sqlite3
Completed scoring steps are reused; this is not a crash-safe wrapper around Gym.train().
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from qagents import (
    Agent,
    AgentRuntime,
    Hooks,
    Limits,
    RunContext,
    RunInterrupted,
    SQLiteStore,
    tool,
)
from qagents.models.testing import FunctionModel, call

from qagent_gym.gym import ExactMatchEvaluator
from qagent_gym.records import Context, Generation, Task

from ._support import run_cli, user_prompt

TASKS = (
    Task("small-sum", json.dumps({"numbers": [2, 3]}), "eval"),
    Task("large-sum", json.dumps({"numbers": [20, 22]}), "eval"),
)
ANSWERS = {"small-sum": "5", "large-sum": "42"}


@dataclass
class EvaluationInputs:
    """Trusted answer keys live outside the model request and are rebuilt on restart."""

    evaluator: ExactMatchEvaluator = field(default_factory=lambda: ExactMatchEvaluator(ANSWERS))
    calls: list[str] = field(default_factory=list)


@tool(read_only=True)
async def score_answer(ctx: RunContext, task_id: str, answer: str) -> str:
    """Score an already generated answer without changing the playbook or memory."""
    inputs: EvaluationInputs = ctx.deps
    task = next(task for task in TASKS if task.task_id == task_id)
    reward = await inputs.evaluator.evaluate(task, Generation(answer, model="offline-arithmetic"))
    inputs.calls.append(task_id)  # Diagnostic counter only.
    return json.dumps({"task_id": task_id, "answer": answer, "reward": reward.value})


@dataclass
class BeforeSecondScore(Hooks):
    """A deterministic deploy boundary after the first tool result was journaled."""

    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def before_tool(self, ctx, tool_call):
        if tool_call.name == "score_answer" and tool_call.arguments["task_id"] == "large-sum":
            self.entered.set()
            await self.release.wait()
        return tool_call


def evaluation_agent(*, pause: BeforeSecondScore | None = None) -> Agent:
    def policy(request):
        completed = [
            json.loads(part.content)
            for message in request.messages
            for part in message.tool_results
            if part.name == "score_answer"
        ]
        done = {item["task_id"] for item in completed}
        tasks = json.loads(user_prompt(request))["tasks"]
        for task in tasks:
            if task["task_id"] not in done:
                answer = str(sum(json.loads(task["prompt"])["numbers"]))
                return call("score_answer", task_id=task["task_id"], answer=answer)
        return json.dumps(completed)

    return Agent(
        FunctionModel(policy),
        name="background-evaluation",
        tools=[score_answer],
        hooks=[pause] if pause is not None else [],
        limits=Limits(max_turns=5, timeout_s=30),
    )


def evaluation_prompt(context: Context) -> str:
    # The complete frozen input is persisted in run_started/message before start returns.
    # Expected answers are intentionally absent from this request.
    return json.dumps(
        {
            "context": context.render(),
            "revision": context.revision,
            "tasks": [{"task_id": task.task_id, "prompt": task.prompt} for task in TASKS],
        }
    )


async def run(path: str | Path, *, namespace: str = "evaluation") -> dict:
    session_id = f"evaluation:{namespace}:{uuid.uuid4().hex}"
    context = Context(namespace)
    pause = BeforeSecondScore()
    first_inputs = EvaluationInputs()
    async with SQLiteStore(path) as store:
        worker = AgentRuntime(
            store,
            [evaluation_agent(pause=pause)],
            worker_id="evaluation-before-deploy",
            deps_factory=lambda session_id: first_inputs,
            session_prefix=session_id,
        )
        try:
            handle = await worker.start(
                "background-evaluation", evaluation_prompt(context), session_id=session_id
            )
            await asyncio.wait_for(pause.entered.wait(), 5)
            original_run_id = handle.run_id
            # Zero grace is deliberate: the hook waits indefinitely at this demo boundary.
            # shutdown leaves the run open; handle.cancel() would permanently close it.
            await worker.shutdown(timeout_s=0)
            try:
                await handle.wait()
            except RunInterrupted:
                pass
            else:
                raise RuntimeError("The demonstration did not interrupt its worker")
            stopped_status = await store.status(session_id)
        finally:
            pause.release.set()
            await worker.shutdown(timeout_s=1)

    # Fresh connection, Agent, dependencies, and worker: nothing relies on the old handle.
    second_inputs = EvaluationInputs()
    async with SQLiteStore(path) as store:
        worker = AgentRuntime(
            store,
            [evaluation_agent()],
            worker_id="evaluation-after-deploy",
            deps_factory=lambda session_id: second_inputs,
            session_prefix=session_id,
        )
        try:
            recovered = await worker.recover()
            if len(recovered) != 1:
                raise RuntimeError(f"Expected one recoverable job, got {len(recovered)}")
            result = await asyncio.wait_for(recovered[0].wait(), 5)
            if not result.completed:
                raise RuntimeError(f"Evaluation stopped: {result.status}")
            return {
                "session_id": session_id,
                "status_after_shutdown": stopped_status.status,
                "lease_released": stopped_status.lease_owner is None,
                "same_run_after_recovery": result.run_id == original_run_id,
                "calls_before_shutdown": first_inputs.calls,
                "calls_after_recovery": second_inputs.calls,
                "scores": json.loads(result.text),
                "final_status": result.status,
                "final_store_status": (await store.status(session_id)).status,
                "frozen_context": context == Context(namespace),
                "recoverable_after_finish": len(await worker.recover()),
            }
        finally:
            await worker.shutdown(timeout_s=1)


if __name__ == "__main__":
    run_cli(run, __doc__)
