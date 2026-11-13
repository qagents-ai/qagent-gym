"""Local fictional banking application driven by actual qagents tool calls.

Run: python -m examples.banking_app --db .gym/banking.sqlite3 demo
The CLI's principal selector simulates an authenticated application session. It
is not authentication. No real accounts, payment rails, network or paid model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import StrictInt
from qagents import Agent, Limits, Permissions, ToolCallPart, ToolError, tool
from qagents.models import ModelRequest
from qagents.models.testing import FunctionModel

from .banking_ledger import BankError, Ledger, Principal


def banking_tools(ledger: Ledger, principal: Principal):
    """Bind identity once in trusted code; the model cannot supply or change it."""
    if not isinstance(principal, Principal):
        raise BankError("an authenticated Principal is required")

    @tool(read_only=True)
    async def list_accounts() -> list[dict]:
        """List only the authenticated customer's fictional accounts and balances."""
        return ledger.accounts(principal)

    @tool(read_only=True)
    async def get_balance(account_id: str) -> dict:
        """Read one owned account balance in integer TEST minor units."""
        try:
            return ledger.balance(principal, account_id)
        except BankError as exc:
            raise ToolError(str(exc)) from exc

    @tool(read_only=True)
    async def get_history(account_id: str, limit: StrictInt = 20) -> list[dict]:
        """Read recent ledger entries for one owned account, newest first."""
        try:
            return ledger.history(principal, account_id, limit=limit)
        except BankError as exc:
            raise ToolError(str(exc)) from exc

    @tool(requires_approval=True, read_only=False)
    async def transfer(
        source_id: str, destination_id: str, amount_minor: StrictInt, idempotency_key: str
    ) -> dict:
        """Transfer fictional funds from an owned account. Reuse a key only for the same intent."""
        try:
            return ledger.transfer(
                principal,
                source_id=source_id,
                destination_id=destination_id,
                amount_minor=amount_minor,
                idempotency_key=idempotency_key,
            )
        except BankError as exc:
            raise ToolError(str(exc)) from exc

    return [list_accounts, get_balance, get_history, transfer]


def banking_policy(request: ModelRequest):
    """A deterministic offline command policy, not a natural-language banking adviser."""
    if request.messages[-1].role == "tool":
        return json.dumps(
            [
                {"tool": part.name, "is_error": part.is_error, "result": part.content}
                for part in request.messages[-1].tool_results
            ]
        )
    command = json.loads(request.messages[-1].text)
    return ToolCallPart(
        id="bank-operation", name=command["operation"], arguments=command.get("arguments", {})
    )


async def run_action(
    ledger: Ledger, principal: Principal, operation: str, arguments: dict | None = None
) -> dict:
    """Execute through Agent, retaining real tool results so refusals remain visible."""
    agent = Agent(
        FunctionModel(banking_policy),
        name="banking-sandbox",
        tools=banking_tools(ledger, principal),
        # The host explicitly permits this sandbox's transfers. Server-side ownership,
        # amount validation, funds and idempotency checks still apply on every call.
        permissions=Permissions(allow=["transfer"]),
    )
    result = await agent.run(
        json.dumps({"operation": operation, "arguments": arguments or {}}),
        limits=Limits(max_turns=3, timeout_s=10, max_total_tokens=4000),
    )
    results = []
    calls = []
    for message in result.new_messages:
        calls.extend(
            {"name": call.name, "arguments": call.arguments} for call in message.tool_calls
        )
        for part in message.tool_results:
            try:
                content = json.loads(part.content)
            except json.JSONDecodeError:
                content = part.content
            results.append({"name": part.name, "is_error": part.is_error, "content": content})
    if result.status != "completed":
        raise RuntimeError(f"sandbox agent did not complete: {result.status}")
    return {"status": result.status, "tool_calls": calls, "tool_results": results}


async def run_demo(ledger: Ledger) -> dict:
    """Idempotent demo: reopening and rerunning never repeats its committed transfer."""
    alice = Principal("alice")
    before = await run_action(ledger, alice, "list_accounts")
    arguments = {
        "source_id": "alice-checking",
        "destination_id": "bob-checking",
        "amount_minor": 1250,
        "idempotency_key": "banking-demo-transfer-v1",
    }
    posted = await run_action(ledger, alice, "transfer", arguments)
    replay = await run_action(ledger, alice, "transfer", arguments)
    refused = await run_action(
        ledger,
        alice,
        "transfer",
        {
            **arguments,
            "source_id": "bob-checking",
            "destination_id": "alice-checking",
            "idempotency_key": "banking-demo-unauthorized-v1",
        },
    )
    return {
        "mode": "fictional local sandbox; no real money or network",
        "accounts_before": before["tool_results"],
        "transfer": posted["tool_results"],
        "replay_matches": posted["tool_results"] == replay["tool_results"],
        "unauthorized_attempt": refused["tool_results"],
        "accounts_after": (await run_action(ledger, alice, "list_accounts"))["tool_results"],
        "history": (
            await run_action(ledger, alice, "get_history", {"account_id": "alice-checking"})
        )["tool_results"],
        "audit": ledger.audit(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(".gym/banking.sqlite3"))
    parser.add_argument(
        "--principal",
        choices=("alice", "bob"),
        default="alice",
        help="simulate a trusted authenticated session; this is not a login",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    commands.add_parser("accounts")
    balance = commands.add_parser("balance")
    balance.add_argument("account_id")
    history = commands.add_parser("history")
    history.add_argument("account_id")
    history.add_argument("--limit", type=int, default=20)
    transfer = commands.add_parser("transfer")
    transfer.add_argument("source_id")
    transfer.add_argument("destination_id")
    transfer.add_argument("amount_minor", type=int)
    transfer.add_argument("--key", required=True)
    args = parser.parse_args(argv)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Ledger(args.db) as ledger:
            ledger.seed_demo()
            if args.command == "demo":
                report = asyncio.run(run_demo(ledger))
            else:
                principal = Principal(args.principal)
                if args.command == "accounts":
                    operation, arguments = "list_accounts", {}
                elif args.command == "balance":
                    operation, arguments = "get_balance", {"account_id": args.account_id}
                elif args.command == "history":
                    operation, arguments = (
                        "get_history",
                        {
                            "account_id": args.account_id,
                            "limit": args.limit,
                        },
                    )
                else:
                    operation, arguments = (
                        "transfer",
                        {
                            "source_id": args.source_id,
                            "destination_id": args.destination_id,
                            "amount_minor": args.amount_minor,
                            "idempotency_key": args.key,
                        },
                    )
                report = asyncio.run(run_action(ledger, principal, operation, arguments))
    except (OSError, BankError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(any(part["is_error"] for part in report.get("tool_results", [])))


if __name__ == "__main__":
    raise SystemExit(main())
