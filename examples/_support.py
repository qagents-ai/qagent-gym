"""Small display/input helpers shared by the offline recipes."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from qagents.models import ModelRequest

from qagent_gym.storage import SQLiteStore


def user_prompt(request: ModelRequest) -> str:
    return next(message.text for message in reversed(request.messages) if message.role == "user")


def task_payload(request: ModelRequest) -> dict[str, Any]:
    return json.loads(user_prompt(request).rsplit("TASK:\n", 1)[1])


def has_rule(request: ModelRequest, rule: str) -> bool:
    # Inspect only the playbook, so retrieved feedback cannot accidentally serve
    # as the learned rule in these controlled integration demonstrations.
    playbook = user_prompt(request).split("\n\nRetrieved memory:", 1)[0]
    return rule in playbook


def state(store: SQLiteStore, namespace: str) -> tuple:
    return store.snapshot(namespace), store.memories(namespace), store.changes(namespace)


def run_cli(run: Callable[..., Awaitable[dict[str, Any]]], description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--db", type=Path, help="Keep SQLite state at this path; default is temporary"
    )
    parser.add_argument("--namespace", default="example")
    args = parser.parse_args()
    try:
        if args.db is None:
            with TemporaryDirectory(prefix="qagent-gym-example-") as directory:
                report = asyncio.run(run(Path(directory) / "gym.sqlite3", namespace=args.namespace))
        else:
            args.db.parent.mkdir(parents=True, exist_ok=True)
            report = asyncio.run(run(args.db, namespace=args.namespace))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"mode": "offline scripted demonstration", **report}, indent=2))
