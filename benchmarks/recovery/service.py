"""Controller-owned loopback effect service with an independent durable ledger."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .protocol import AMOUNT, marker


class PayloadConflict(ValueError):
    """An idempotency key was reused for a different intent."""


class Ledger:
    def __init__(self, path: Path, *, deduplicate: bool) -> None:
        self.path = path
        self.deduplicate = deduplicate
        with self.connect() as connection:
            connection.executescript(
                "CREATE TABLE credits (id INTEGER PRIMARY KEY, key TEXT NOT NULL, "
                "amount INTEGER NOT NULL);"
                "CREATE TABLE intents (key TEXT PRIMARY KEY, amount INTEGER NOT NULL, "
                "credit_id INTEGER NOT NULL REFERENCES credits(id));"
                "CREATE TABLE requests (id INTEGER PRIMARY KEY, key TEXT NOT NULL, "
                "amount INTEGER NOT NULL, phase TEXT NOT NULL, credit_id INTEGER NOT NULL);"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def credit(self, key: str, amount: int, phase: str) -> dict:
        if not isinstance(key, str) or not key or len(key) > 200:
            raise ValueError("invalid operation key")
        if type(amount) is not int or amount <= 0 or phase not in ("first", "recovery"):
            raise ValueError("invalid credit request")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT amount, credit_id FROM intents WHERE key=?", (key,)
            ).fetchone()
            if existing and existing[0] != amount:
                raise PayloadConflict("same key with different amount")
            if existing and self.deduplicate:
                credit_id = existing[1]
            else:
                cursor = connection.execute(
                    "INSERT INTO credits(key, amount) VALUES (?, ?)", (key, amount)
                )
                credit_id = cursor.lastrowid
                if not existing:
                    connection.execute(
                        "INSERT INTO intents VALUES (?, ?, ?)", (key, amount, credit_id)
                    )
            connection.execute(
                "INSERT INTO requests(key, amount, phase, credit_id) VALUES (?, ?, ?, ?)",
                (key, amount, phase, credit_id),
            )
        return {"key": key, "amount": amount, "credit_id": credit_id}

    def snapshot(self) -> dict:
        with self.connect() as connection:
            credits = connection.execute(
                "SELECT id, key, amount FROM credits ORDER BY id"
            ).fetchall()
            requests = connection.execute(
                "SELECT id, key, amount, phase, credit_id FROM requests ORDER BY id"
            ).fetchall()
        return {
            "credits": [dict(zip(("id", "key", "amount"), row, strict=True)) for row in credits],
            "requests": [
                dict(zip(("id", "key", "amount", "phase", "credit_id"), row, strict=True))
                for row in requests
            ],
            "credit_count": len(credits),
            "request_count": len(requests),
            "balance": sum(row[2] for row in credits),
        }


class EffectService:
    def __init__(self, directory: Path, *, fault: str, deduplicate: bool, timeout: float) -> None:
        self.directory = directory
        self.fault = fault
        self.timeout = timeout
        self.ledger = Ledger(directory / "service.sqlite3", deduplicate=deduplicate)
        self.release = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def handler(self) -> type[BaseHTTPRequestHandler]:
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_POST(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if self.path != "/credit" or not 0 < length <= 4096:
                        raise ValueError("invalid request envelope")
                    body = json.loads(self.rfile.read(length))
                    if set(body) != {"key", "amount", "phase"} or body["amount"] != AMOUNT:
                        raise ValueError("invalid workload")
                    receipt = service.ledger.credit(body["key"], body["amount"], body["phase"])
                except PayloadConflict:
                    self.send_error(409, "conflicting operation")
                    return
                except (ValueError, KeyError, TypeError):
                    self.send_error(400, "invalid request")
                    return
                if service.fault == "after_commit_before_ack" and body["phase"] == "first":
                    marker(
                        service.directory / "barrier.json",
                        service.fault,
                        service_commit=receipt,
                    )
                    # The controller kills the worker while this response is withheld.
                    service.release.wait(service.timeout)
                payload = json.dumps(receipt).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        return Handler

    def __enter__(self) -> EffectService:
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
