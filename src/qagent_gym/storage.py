"""SQLite persistence with scoped memory and atomic, idempotent playbook changes."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from .records import Bullet, Context, Episode, MemoryRecord, PlaybookDelta, _text


class SQLiteStore:
    """One connection per owning thread; separate connections serialize write batches.

    Use as a context manager to release database handles on Windows. Memory search
    ranks by case-insensitive word overlap, then memory_id; it does not use embeddings.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(str(path), timeout=30)
        self._db.row_factory = sqlite3.Row
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self._db.close()
            raise ValueError(f"unsupported database schema version: {version}")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                namespace TEXT NOT NULL, memory_id TEXT NOT NULL, text TEXT NOT NULL,
                source_episode_id TEXT NOT NULL, PRIMARY KEY(namespace, memory_id)
            );
            CREATE TABLE IF NOT EXISTS revisions (
                namespace TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS bullets (
                namespace TEXT NOT NULL, bullet_id TEXT NOT NULL, text TEXT NOT NULL,
                source_episode_id TEXT NOT NULL, helpful INTEGER NOT NULL,
                harmful INTEGER NOT NULL, active INTEGER NOT NULL,
                PRIMARY KEY(namespace, bullet_id)
            );
            CREATE TABLE IF NOT EXISTS deltas (
                namespace TEXT NOT NULL, delta_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(namespace, delta_id)
            );
            CREATE TABLE IF NOT EXISTS episodes (
                namespace TEXT NOT NULL, run_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(namespace, run_id)
            );
            PRAGMA user_version = 1;
        """)

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def add(self, record: MemoryRecord) -> None:
        """Insert memory; an identical retry is harmless, changed identity is rejected."""
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT text, source_episode_id FROM memories WHERE namespace=? AND memory_id=?",
                (record.namespace, record.memory_id),
            ).fetchone()
            if row:
                if (row["text"], row["source_episode_id"]) != (
                    record.text,
                    record.source_episode_id,
                ):
                    raise ValueError("memory_id reused with a different payload")
                return
            self._db.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?)",
                (record.namespace, record.memory_id, record.text, record.source_episode_id),
            )

    def search(self, query: str, *, namespace: str, limit: int = 5) -> tuple[MemoryRecord, ...]:
        _text(namespace, "namespace")
        _text(query, "query", empty=True)
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        words = set(re.findall(r"\w+", query.casefold()))
        ranked = []
        for row in self._db.execute("SELECT * FROM memories WHERE namespace=?", (namespace,)):
            score = len(words.intersection(re.findall(r"\w+", row["text"].casefold())))
            if score:
                ranked.append((score, MemoryRecord(**dict(row))))
        ranked.sort(key=lambda item: (-item[0], item[1].memory_id))
        return tuple(record for _, record in ranked[:limit])

    def delete(self, memory_id: str, *, namespace: str) -> None:
        """Delete one scoped record; absence is a no-op."""
        with self._db:
            self._db.execute(
                "DELETE FROM memories WHERE namespace=? AND memory_id=?", (namespace, memory_id)
            )

    def memories(self, namespace: str) -> tuple[MemoryRecord, ...]:
        return tuple(
            MemoryRecord(**dict(row))
            for row in self._db.execute(
                "SELECT * FROM memories WHERE namespace=? ORDER BY memory_id", (namespace,)
            )
        )

    def snapshot(self, namespace: str) -> Context:
        """Read a consistent immutable playbook, including deprecated bullet identities."""
        _text(namespace, "namespace")
        with self._db:
            self._db.execute("BEGIN")
            row = self._db.execute(
                "SELECT revision FROM revisions WHERE namespace=?", (namespace,)
            ).fetchone()
            bullets = tuple(
                Bullet(
                    bullet_id=item["bullet_id"],
                    text=item["text"],
                    source_episode_id=item["source_episode_id"],
                    helpful=item["helpful"],
                    harmful=item["harmful"],
                    active=bool(item["active"]),
                )
                for item in self._db.execute(
                    "SELECT * FROM bullets WHERE namespace=? ORDER BY bullet_id", (namespace,)
                )
            )
        return Context(namespace=namespace, revision=row[0] if row else 0, bullets=bullets)

    def apply(self, namespace: str, changes: Iterable[PlaybookDelta]) -> int:
        """Apply an entire batch or nothing. Return the number of newly applied deltas."""
        _text(namespace, "namespace")
        changes = tuple(changes)
        if not all(isinstance(change, PlaybookDelta) for change in changes):
            raise ValueError("changes must contain PlaybookDelta values")
        applied = 0
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            for change in changes:
                payload = json.dumps(asdict(change), sort_keys=True, ensure_ascii=False)
                prior = self._db.execute(
                    "SELECT payload FROM deltas WHERE namespace=? AND delta_id=?",
                    (namespace, change.delta_id),
                ).fetchone()
                if prior:
                    if prior[0] != payload:
                        raise ValueError("delta_id reused with a different payload")
                    continue
                bullet = self._db.execute(
                    "SELECT * FROM bullets WHERE namespace=? AND bullet_id=?",
                    (namespace, change.bullet_id),
                ).fetchone()
                if change.operation == "add":
                    if bullet:
                        raise ValueError(f"bullet already exists: {change.bullet_id}")
                    self._db.execute(
                        "INSERT INTO bullets VALUES (?, ?, ?, ?, ?, ?, 1)",
                        (
                            namespace,
                            change.bullet_id,
                            change.text,
                            change.source_episode_id,
                            change.helpful,
                            change.harmful,
                        ),
                    )
                else:
                    if not bullet:
                        raise ValueError(f"unknown bullet: {change.bullet_id}")
                    self._db.execute(
                        """
                        UPDATE bullets SET text=?, source_episode_id=?, helpful=helpful+?,
                        harmful=harmful+?, active=? WHERE namespace=? AND bullet_id=?
                    """,
                        (
                            change.text if change.operation == "update" else bullet["text"],
                            change.source_episode_id,
                            change.helpful,
                            change.harmful,
                            int(change.operation == "update"),
                            namespace,
                            change.bullet_id,
                        ),
                    )
                self._db.execute(
                    "INSERT INTO deltas VALUES (?, ?, ?)", (namespace, change.delta_id, payload)
                )
                applied += 1
            if applied:
                self._db.execute(
                    """
                    INSERT INTO revisions VALUES (?, ?)
                    ON CONFLICT(namespace) DO UPDATE SET revision=revision+excluded.revision
                """,
                    (namespace, applied),
                )
        return applied

    def changes(self, namespace: str) -> tuple[PlaybookDelta, ...]:
        return tuple(
            PlaybookDelta(**json.loads(row[0]))
            for row in self._db.execute(
                "SELECT payload FROM deltas WHERE namespace=? ORDER BY rowid", (namespace,)
            )
        )

    def save_episode(self, episode: Episode) -> None:
        payload = episode.to_json()
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT payload FROM episodes WHERE namespace=? AND run_id=?",
                (episode.context.namespace, episode.run_id),
            ).fetchone()
            if row:
                if row[0] != payload:
                    raise ValueError("run_id reused with a different episode")
                return
            self._db.execute(
                "INSERT INTO episodes VALUES (?, ?, ?)",
                (episode.context.namespace, episode.run_id, payload),
            )

    def episodes(self, namespace: str) -> tuple[Episode, ...]:
        return tuple(
            Episode.from_json(row[0])
            for row in self._db.execute(
                "SELECT payload FROM episodes WHERE namespace=? ORDER BY rowid", (namespace,)
            )
        )
