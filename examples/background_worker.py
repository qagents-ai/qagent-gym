"""Queue local report jobs, cancel one, and reopen their durable status.

Run: python -m examples.background_worker --db reports.sqlite3
Offline FunctionModel, one-host SQLite, and read-only fixture data only.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from qagents import Agent, AgentRuntime, Limits, RunContext, SQLiteStore, tool
from qagents.models.testing import FunctionModel, call

from ._support import run_cli, user_prompt


@dataclass
class ReportInputs:
    """Rebuild process-local clients and synchronization in each worker."""

    records: dict[str, str]
    calls: list[str] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event | None = None


@tool(read_only=True)
async def read_report_source(ctx: RunContext, report_id: str) -> str:
    """Read a report's source from the worker's fixture catalogue."""
    inputs: ReportInputs = ctx.deps
    inputs.calls.append(report_id)  # Diagnostic counter, not application state.
    inputs.started.set()
    if inputs.release is not None:
        await inputs.release.wait()
    return inputs.records[report_id]


def report_agent() -> Agent:
    def policy(request):
        if request.messages[-1].role == "tool":
            return "Report: " + request.messages[-1].tool_results[0].content
        return call("read_report_source", report_id=json.loads(user_prompt(request))["report_id"])

    return Agent(
        FunctionModel(policy),
        name="background-report",
        tools=[read_report_source],
        limits=Limits(max_turns=3, timeout_s=30),
    )


async def run(path: str | Path, *, namespace: str = "reports") -> dict:
    prefix = f"reports:{namespace}:{uuid.uuid4().hex}:"
    inputs = ReportInputs(
        {"first": "Evidence collected.", "second": "Budget checked.", "cancelled": "Unused."},
        release=asyncio.Event(),
    )
    session_ids = {name: prefix + name for name in inputs.records}
    async with SQLiteStore(path) as store:
        worker = AgentRuntime(
            store,
            [report_agent()],
            max_concurrent=1,
            deps_factory=lambda session_id: inputs,
            session_prefix=prefix,
        )
        try:
            first = await worker.start(
                "background-report",
                json.dumps({"report_id": "first"}),
                session_id=session_ids["first"],
            )
            await asyncio.wait_for(inputs.started.wait(), 5)
            second = await worker.start(
                "background-report",
                json.dumps({"report_id": "second"}),
                session_id=session_ids["second"],
            )
            cancelled = await worker.start(
                "background-report",
                json.dumps({"report_id": "cancelled"}),
                session_id=session_ids["cancelled"],
            )
            accepted = {
                name: (await store.status(session_id)).status
                for name, session_id in session_ids.items()
            }
            queued = {"second": second.status, "cancelled": cancelled.status}
            cancelled.cancel("No longer needed")
            cancelled_result = await asyncio.wait_for(cancelled.wait(), 5)
            inputs.release.set()
            first_result, second_result = await asyncio.wait_for(
                asyncio.gather(first.wait(), second.wait()), 5
            )
            report = {
                "session_ids": session_ids,
                "accepted_store_status": accepted,
                "queued_handle_status": queued,
                "results": {
                    "first": first_result.status,
                    "second": second_result.status,
                    "cancelled": cancelled_result.status,
                },
                "outputs": [first_result.text, second_result.text],
                "cancelled_model_requests": cancelled_result.usage.requests,
                "source_reads": list(inputs.calls),
            }
        finally:
            inputs.release.set()
            await worker.shutdown(timeout_s=1)
    # A new connection reads the journal; handles belong only to the old worker.
    async with SQLiteStore(path) as store:
        report["reopened_store_status"] = {
            name: (await store.status(session_id)).status
            for name, session_id in session_ids.items()
        }
        worker = AgentRuntime(store, [report_agent()], session_prefix=prefix)
        try:
            report["recoverable_after_finish"] = len(await worker.recover())
        finally:
            await worker.shutdown(timeout_s=1)
    return report


if __name__ == "__main__":
    run_cli(run, __doc__)
