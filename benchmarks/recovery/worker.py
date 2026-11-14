"""One owned subprocess: execute or recover one framework's identical credit intent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import TypedDict

from .protocol import event, marker, read_json, request_effect, write_json


async def async_barrier(config: dict, point: str) -> None:
    if config["phase"] == "first" and config["fault"] == point:
        marker(Path(config["directory"]) / "barrier.json", point)
        while True:
            await asyncio.sleep(0.01)


def sync_barrier(config: dict, point: str) -> None:
    if config["phase"] == "first" and config["fault"] == point:
        marker(Path(config["directory"]) / "barrier.json", point)
        while True:
            time.sleep(0.01)


async def qagents_work(config: dict) -> dict:
    from qagents import Agent, AgentRuntime, SQLiteStore, ToolCallPart, tool
    from qagents.models.testing import FunctionModel

    directory = Path(config["directory"])
    phase = config["phase"]
    key = config["key"]

    @tool(idempotent=True)
    async def credit_operation(key: str) -> str:
        """Credit 10 TEST units using a durable business key honored by the service."""
        event(directory, phase, "effect_entered", key=key)
        await async_barrier(config, "before_effect")
        receipt = await asyncio.to_thread(
            request_effect, config["service_url"], key, phase, config["timeout"]
        )
        event(directory, phase, "effect_returned", receipt=receipt)
        return json.dumps(receipt, sort_keys=True)

    async def policy(request):
        results = [part for message in request.messages for part in message.tool_results]
        if results:
            if len(results) != 1 or results[0].is_error:
                raise ValueError("unexpected tool outcome")
            receipt = json.loads(results[0].content)
            event(directory, phase, "finalize_entered", receipt=receipt)
            await async_barrier(config, "after_checkpoint")
            return json.dumps(receipt, sort_keys=True)
        event(directory, phase, "plan")
        return ToolCallPart(id="credit-call", name="credit_operation", arguments={"key": key})

    agent = Agent(FunctionModel(policy), tools=[credit_operation], name="credit-agent")
    store = SQLiteStore(directory / "framework.sqlite3")
    runtime = AgentRuntime(
        store,
        [agent],
        lease_ttl_s=config["lease_ttl"],
        heartbeat_s=config["lease_ttl"] / 3,
        max_concurrent=1,
    )
    marker(directory / f"{phase}.ready.json", "ready", framework="qagents")
    started = time.perf_counter_ns()
    lease_wait_ms = 0.0
    try:
        if phase == "first":
            handle = await runtime.start(agent.name, key, session_id=key)
        else:
            entries = await store.load(key)
            results = [entry.data["result"] for entry in entries if entry.kind == "tool_result"]
            status = await store.status(key)
            write_json(
                directory / "pre_recovery.json",
                {
                    "tool_result_count": len(results),
                    "tool_results": results,
                    "status": status.status,
                    "journal_kinds": [entry.kind for entry in entries],
                    "lease_expires_at": status.lease_expires_at,
                },
            )
            wait_started = time.perf_counter_ns()
            deadline = time.monotonic() + config["timeout"]
            while not (handles := await runtime.recover()):
                if time.monotonic() >= deadline:
                    raise TimeoutError("runtime did not acquire a recoverable session")
                await asyncio.sleep(0.01)
            lease_wait_ms = (time.perf_counter_ns() - wait_started) / 1e6
            if len(handles) != 1 or handles[0].session_id != key:
                raise ValueError("recovery selected an unexpected session")
            handle = handles[0]
        result = await handle.wait()
        if result.status != "completed":
            raise ValueError(f"run was not completed: {result.status}")
        receipt = json.loads(result.output)
        return {
            "receipt": receipt,
            "framework_status": result.status,
            "work_ms": (time.perf_counter_ns() - started) / 1e6,
            "lease_acquisition_wait_ms": lease_wait_ms,
        }
    finally:
        await runtime.shutdown()
        await store.aclose()


async def effects_work(config: dict) -> dict:
    from pydantic import BaseModel
    from qagents import Agent, RunContext, Session, SQLiteStore, ToolCallPart, tool
    from qagents.effects import Attempt, Dedupe, EffectContext, Key, effect
    from qagents.models.testing import FunctionModel

    directory = Path(config["directory"])
    phase, key = config["phase"], config["key"]

    class CreditRequest(BaseModel):
        key: str

    class CreditReceipt(BaseModel):
        key: str
        amount: int
        credit_id: int

    @effect(
        "benchmark.credit",
        key=Key("credit", "key"),
        dedupe=Dedupe(3600, contract="local SQLite fixture", concurrent="waits"),
        timeout_s=2,
    )
    async def credit(ctx: EffectContext, attempt: Attempt[CreditRequest]) -> CreditReceipt:
        event(directory, phase, "effect_entered", key=attempt.request.key)
        await async_barrier(config, "before_effect")
        receipt = await asyncio.to_thread(
            request_effect,
            config["service_url"],
            attempt.request.key,
            phase,
            config["timeout"],
            idempotency_key=attempt.idempotency_key,
        )
        event(directory, phase, "effect_returned", receipt=receipt)
        return CreditReceipt.model_validate(receipt)

    async def credit_operation(ctx, key: str) -> str:
        """Credit 10 TEST units through the destination's idempotency contract."""
        return (await ctx.perform(credit, CreditRequest(key=key))).model_dump_json()

    credit_operation.__annotations__["ctx"] = RunContext
    credit_tool = tool(effects=[credit])(credit_operation)

    async def policy(request):
        results = [part for message in request.messages for part in message.tool_results]
        if not results:
            return ToolCallPart(id="credit-call", name="credit_operation", arguments={"key": key})
        if len(results) != 1 or results[0].is_error:
            raise ValueError("unexpected tool outcome")
        receipt = json.loads(results[0].content)
        event(directory, phase, "finalize_entered", receipt=receipt)
        await async_barrier(config, "after_checkpoint")
        return json.dumps(receipt, sort_keys=True)

    # The lease needs to cover the send timeout and both safety margins.
    ttl = config["lease_ttl"]
    agent = Agent(FunctionModel(policy), tools=[credit_tool], lease_ttl_s=ttl)
    marker(directory / f"{phase}.ready.json", "ready", framework="qagents-effects")
    started = time.perf_counter_ns()
    lease_wait_ms = 0.0
    async with SQLiteStore(directory / "framework.sqlite3") as store:
        session = Session(store, id=key)
        if phase == "first":
            result = await agent.run(key, session=session)
        else:
            entries = await session.entries()
            results = [entry.data["result"] for entry in entries if entry.kind == "tool_result"]
            status = await store.status(key)
            write_json(
                directory / "pre_recovery.json",
                {
                    "tool_result_count": len(results),
                    "tool_results": results,
                    "status": status.status,
                    "journal_kinds": [entry.kind for entry in entries],
                    "lease_expires_at": status.lease_expires_at,
                    "effect_intents": [e.data for e in entries if e.kind == "effect_intent"],
                },
            )
            wait_started = time.perf_counter_ns()
            deadline = time.monotonic() + config["timeout"]
            while status.lease_owner and (
                await store.now() < status.lease_expires_at + store.takeover_grace_s
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Effects lease did not expire")
                await asyncio.sleep(0.01)
                status = await store.status(key)
            lease_wait_ms = (time.perf_counter_ns() - wait_started) / 1e6
            result = await agent.resume(session)
        if result.status != "completed":
            raise ValueError(f"run was not completed: {result.status}")
        entries = await session.entries()
        return {
            "receipt": json.loads(result.output),
            "framework_status": result.status,
            "work_ms": (time.perf_counter_ns() - started) / 1e6,
            "lease_acquisition_wait_ms": lease_wait_ms,
            "effect_intents": [e.data for e in entries if e.kind == "effect_intent"],
            "effect_commits": [e.data for e in entries if e.kind == "effect_committed"],
        }


class CreditState(TypedDict, total=False):
    key: str
    receipt: dict
    output: dict


def langgraph_work(config: dict) -> dict:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    directory = Path(config["directory"])
    phase = config["phase"]

    def plan(state: CreditState) -> dict:
        event(directory, phase, "plan")
        return {"key": state["key"]}

    def credit_operation(state: CreditState) -> dict:
        event(directory, phase, "effect_entered", key=state["key"])
        sync_barrier(config, "before_effect")
        receipt = request_effect(config["service_url"], state["key"], phase, config["timeout"])
        event(directory, phase, "effect_returned", receipt=receipt)
        return {"receipt": receipt}

    def finalize(state: CreditState) -> dict:
        event(directory, phase, "finalize_entered", receipt=state["receipt"])
        sync_barrier(config, "after_checkpoint")
        return {"output": state["receipt"]}

    with closing(
        sqlite3.connect(directory / "framework.sqlite3", check_same_thread=False)
    ) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        checkpointer = SqliteSaver(connection)
        checkpointer.setup()
        graph = (
            StateGraph(CreditState)
            .add_node("plan", plan)
            .add_node("credit", credit_operation)
            .add_node("finalize", finalize)
            .add_edge(START, "plan")
            .add_edge("plan", "credit")
            .add_edge("credit", "finalize")
            .add_edge("finalize", END)
            .compile(checkpointer=checkpointer)
        )
        invocation = {"configurable": {"thread_id": config["key"]}}
        marker(directory / f"{phase}.ready.json", "ready", framework="langgraph")
        started = time.perf_counter_ns()
        if phase == "recovery":
            previous = graph.get_state(invocation)
            write_json(
                directory / "pre_recovery.json",
                {
                    "tool_result_count": int("receipt" in previous.values),
                    "tool_results": [previous.values["receipt"]]
                    if "receipt" in previous.values
                    else [],
                    "next": list(previous.next),
                },
            )
        result = graph.invoke(
            {"key": config["key"]} if phase == "first" else None,
            invocation,
            durability="sync",
            version="v2",
        )
        final_state = graph.get_state(invocation)
        if result.interrupts or final_state.next:
            raise ValueError("graph did not finish")
        return {
            "receipt": result.value["output"],
            "framework_status": "completed",
            "work_ms": (time.perf_counter_ns() - started) / 1e6,
            "lease_acquisition_wait_ms": 0.0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    if config["framework"] == "qagents-effects":
        result = asyncio.run(effects_work(config))
    elif config["framework"] == "qagents":
        result = asyncio.run(qagents_work(config))
    else:
        result = langgraph_work(config)
    marker(Path(config["directory"]) / f"{config['phase']}.result.json", "completed", **result)


if __name__ == "__main__":
    main()
