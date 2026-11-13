"""Small, durable controller/worker protocol shared by both implementations."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

FAULTS = ("none", "before_effect", "after_commit_before_ack", "after_checkpoint")
FRAMEWORKS = ("qagents", "langgraph")
AMOUNT = 10


def write_json(path: Path, value: Any) -> None:
    """Publish a complete JSON document without replacing an existing artifact."""
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # Readers only consume after the process exits, or tolerate an incomplete marker.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def marker(path: Path, name: str, **fields: Any) -> None:
    write_json(
        path,
        {"name": name, "pid": os.getpid(), "monotonic_ns": time.perf_counter_ns(), **fields},
    )


def event(directory: Path, phase: str, name: str, **fields: Any) -> None:
    record = {"name": name, "pid": os.getpid(), "monotonic_ns": time.perf_counter_ns(), **fields}
    with (directory / f"{phase}.events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def request_effect(url: str, key: str, phase: str, timeout: float) -> dict:
    body = json.dumps({"key": key, "amount": AMOUNT, "phase": phase}).encode()
    request = Request(url + "/credit", data=body, headers={"Content-Type": "application/json"})
    # Loopback traffic must not accidentally use a caller's proxy configuration.
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        receipt = json.loads(response.read(8192))
    if (
        not isinstance(receipt, dict)
        or receipt.get("key") != key
        or type(receipt.get("amount")) is not int
        or receipt["amount"] != AMOUNT
        or type(receipt.get("credit_id")) is not int
        or receipt["credit_id"] < 1
    ):
        raise ValueError("invalid service receipt")
    return receipt


def source_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((source / "qagents").rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def clean_environment(source: Path | None, temporary: Path) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().endswith(("KEY", "TOKEN", "SECRET", "PASSWORD"))
        and not name.upper().startswith(("LANGSMITH_", "LANGCHAIN_", "OTEL_"))
    }
    root = Path(__file__).resolve().parents[2]
    env.update(
        PYTHONPATH=os.pathsep.join(str(path) for path in (root, source) if path is not None),
        PYTHONDONTWRITEBYTECODE="1",
        DO_NOT_TRACK="true",
        MLFLOW_DISABLE_TELEMETRY="true",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        TEMP=str(temporary),
        TMP=str(temporary),
        PYTHONUNBUFFERED="1",
    )
    return env
