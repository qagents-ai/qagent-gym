"""Run tools, durable approval, or a supervisor with an explicitly configured model.

Without --fixture this uses OpenAIChatModel against the configured endpoint.
No model is selected implicitly. Local notes are the only writable application data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import StrictInt
from qagents import Agent, Limits, RunContext, Session, SQLiteStore, ToolError, tool
from qagents.models import OpenAIChatModel
from qagents.models.testing import FunctionModel, call

CATALOG = {"desk-lamp": 2500, "notebook": 500}
DEFAULT_PROMPTS = {
    "tools": "Use the catalog tool to quote two desk lamps. State the total in TEST minor units.",
    "approval": "Save a local note saying: Review the desk-lamp quote before ordering.",
    "supervisor": "Ask the catalog specialist to quote two desk lamps, then summarize its result.",
}


class Notes:
    """Local application store with atomic operation-key deduplication, not a payment service."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS approved_notes "
            "(operation_id TEXT PRIMARY KEY, text TEXT NOT NULL)"
        )
        self.connection.commit()

    def save(self, operation_id: str, text: str) -> dict:
        if not operation_id or not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ValueError("a stable operation ID and a note of 1–500 characters are required")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO approved_notes VALUES (?, ?)", (operation_id, text)
            )
            stored = self.connection.execute(
                "SELECT text FROM approved_notes WHERE operation_id = ?", (operation_id,)
            ).fetchone()[0]
            if stored != text:
                raise ValueError("operation ID already belongs to different note content")
        return {"operation_id": operation_id, "text": stored}

    def count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM approved_notes").fetchone()[0]

    def close(self) -> None:
        self.connection.close()


def tools_for(notes: Notes, executed: list[str]):
    @tool(read_only=True)
    async def quote_catalog(product: str, quantity: StrictInt) -> dict:
        """Quote a fictional catalog product in integer TEST minor units; quantity must be 1–20."""
        if product not in CATALOG or not 1 <= quantity <= 20:
            raise ToolError("Choose desk-lamp or notebook and a quantity from 1 to 20.")
        executed.append("quote_catalog")
        return {
            "product": product,
            "quantity": quantity,
            "total_minor": CATALOG[product] * quantity,
        }

    @tool(requires_approval=True, idempotent=True)
    async def save_note(ctx: RunContext, text: str) -> dict:
        """Save one local review note after approval; no message or order is sent."""
        try:
            result = notes.save(ctx.operation_id, text)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        executed.append("save_note")
        return result

    return quote_catalog, save_note


def fixture_factory(role: str):
    """Explicit offline test policy; never substituted for a configured provider."""

    def policy(request):
        if request.messages[-1].role == "tool":
            part = request.messages[-1].tool_results[0]
            return "Tool refused; nothing else was done." if part.is_error else part.content
        if role == "approval":
            return call("save_note", text="Review the desk-lamp quote before ordering.")
        if role == "supervisor":
            return call("catalog_specialist", task="Quote two desk lamps in TEST minor units.")
        return call("quote_catalog", product="desk-lamp", quantity=2)

    return FunctionModel(policy, name=f"explicit-fixture-{role}")


def build_agent(mode: str, model_factory: Callable, notes: Notes, executed: list[str]) -> Agent:
    quote, save = tools_for(notes, executed)
    limits = Limits(
        max_turns=8,
        max_total_tokens=6000,
        max_tool_calls=5,
        timeout_s=60,
        tool_timeout_s=20,
        max_depth=2,
        model_retries=0,
    )
    if mode == "supervisor":
        specialist = Agent(
            model_factory("specialist"),
            name="catalog-specialist",
            instructions="Use quote_catalog for prices. Report its values; never invent prices.",
            tools=[quote],
            limits=limits,
        )
        return Agent(
            model_factory("supervisor"),
            name="catalog-supervisor",
            instructions="Delegate calculations to catalog_specialist; summarize its answer.",
            tools=[specialist.as_tool(name="catalog_specialist", read_only=True, timeout=30)],
            limits=limits,
        )
    return Agent(
        model_factory(mode),
        name=f"real-{mode}",
        instructions=(
            "Use save_note only when the user requests a local note. Do not claim it was saved "
            "until the tool succeeds. Respect a denied approval."
            if mode == "approval"
            else "Use quote_catalog for prices and totals. Do not invent prices or place orders."
        ),
        tools=[save] if mode == "approval" else [quote],
        limits=limits,
    )


async def run_scenario(
    model_factory: Callable,
    mode: str,
    database: Path,
    *,
    prompt: str | None = None,
    session_id: str | None = None,
    decision: bool | None = None,
    tool_call_id: str | None = None,
) -> dict:
    """Run or explicitly resume; tool decisions always come from the host, not the model."""
    if mode not in {"tools", "approval", "supervisor", "resume"}:
        raise ValueError("unknown scenario")
    if mode == "resume":
        if (
            not session_id
            or not session_id.startswith("real-approval:")
            or type(decision) is not bool
            or not tool_call_id
        ):
            raise ValueError(
                "resume requires an approval session, explicit decision and tool call ID"
            )
        if prompt is not None:
            raise ValueError("resume continues the recorded prompt; it cannot accept a replacement")
    elif session_id is not None or decision is not None or tool_call_id is not None:
        raise ValueError("only resume accepts a recorded session or approval decision")
    if prompt is not None and (not prompt.strip() or len(prompt) > 2000):
        raise ValueError("prompt must contain 1–2000 characters")
    database = Path(database)
    if mode == "resume" and not database.is_file():
        raise ValueError("resume requires the existing session database")
    database.parent.mkdir(parents=True, exist_ok=True)
    notes = Notes(database.with_suffix(database.suffix + ".notes.sqlite3"))
    executed: list[str] = []
    try:
        agent = build_agent(
            "approval" if mode == "resume" else mode, model_factory, notes, executed
        )
        async with SQLiteStore(database) as store:
            session = Session(store, id=session_id or f"real-{mode}:{uuid.uuid4().hex}")
            if mode == "resume":
                state = await session.state()
                if tool_call_id not in {
                    pending.tool_call_id for pending in state.pending_approvals
                }:
                    raise ValueError(
                        "tool call ID does not match a pending approval in this session"
                    )
                result = await agent.resume(session, approvals={tool_call_id: decision})
            else:
                result = await agent.run(prompt or DEFAULT_PROMPTS[mode], session=session)
            return {
                "scenario": mode,
                "session_id": session.id,
                "status": result.status,
                "output": result.output if result.status == "completed" else None,
                "pending_approvals": [
                    {
                        "tool_call_id": pending.tool_call_id,
                        "name": pending.name,
                        "arguments": pending.arguments,
                    }
                    for pending in result.pending_approvals
                ],
                "local_tool_executions": executed,
                "stored_note_count": notes.count(),
                "model_requests": result.usage.requests,
            }
    finally:
        notes.close()


def endpoint_config(base_url: str | None, model: str | None) -> tuple[str, str]:
    if not base_url or not model or not model.strip():
        raise ValueError("configure --base-url and --model, or explicitly choose --fixture")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(ord(char) < 33 for char in base_url)
        or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise ValueError("use HTTPS, or HTTP on loopback; put credentials in QAGENTS_API_KEY")
    return base_url.rstrip("/"), model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("tools", "approval", "supervisor", "resume"))
    parser.add_argument(
        "--fixture", action="store_true", help="explicit offline scripted test mode"
    )
    parser.add_argument("--base-url", default=os.environ.get("QAGENTS_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("QAGENTS_MODEL"))
    parser.add_argument("--db", type=Path, default=Path(".gym/real-agents.sqlite3"))
    parser.add_argument("--prompt")
    parser.add_argument("--session")
    parser.add_argument("--call-id")
    parser.add_argument("--decision", choices=("approve", "deny"))
    parser.add_argument(
        "--no-stream-usage", action="store_true", help="for servers without stream_options"
    )
    args = parser.parse_args(argv)
    if args.fixture and (args.base_url or args.model):
        parser.error("--fixture cannot be combined with provider configuration")
    try:
        if not args.fixture:
            base_url, model_id = endpoint_config(args.base_url, args.model)
    except ValueError as exc:
        parser.error(str(exc))

    async def execute():
        options = {
            "prompt": args.prompt,
            "session_id": args.session,
            "decision": None if args.decision is None else args.decision == "approve",
            "tool_call_id": args.call_id,
        }
        if args.fixture:
            return await run_scenario(fixture_factory, args.scenario, args.db, **options)
        async with httpx.AsyncClient(timeout=30) as client:
            model = OpenAIChatModel(
                model_id,
                base_url=base_url,
                api_key=os.environ.get("QAGENTS_API_KEY"),
                http_client=client,
                timeout=30,
                max_retries=0,
                include_usage=not args.no_stream_usage,
            )
            return await run_scenario(lambda role: model, args.scenario, args.db, **options)

    try:
        report = asyncio.run(execute())
    except ValueError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error("agent execution failed; inspect local configuration and provider health")
    report["execution_mode"] = "explicit offline fixture" if args.fixture else "configured provider"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] in {"completed", "paused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
