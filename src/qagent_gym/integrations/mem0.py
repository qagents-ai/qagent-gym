"""Mem0 SDK result normalization and explicit evaluation snapshots."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from qagent_gym.records import MemoryRecord


class Mem0Client(Protocol):
    """The synchronous client methods used by :class:`Mem0Memory`."""

    def add(
        self, messages: str, *, user_id: str, metadata: dict[str, str], infer: bool
    ) -> object: ...

    def search(self, query: str, *, filters: dict[str, str], top_k: int) -> object: ...


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Mem0 {field} must be a nonempty string")
    return value


def _namespace(value: object) -> str:
    namespace = _text(value, "namespace")
    if namespace != namespace.strip():
        raise ValueError("Mem0 namespace must not have leading or trailing whitespace")
    return namespace


def _results(payload: object) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping) or "status" in payload:
        raise ValueError("Mem0 must return committed results, not a pending acknowledgement")
    results = payload.get("results")
    if not isinstance(results, list) or any(not isinstance(item, Mapping) for item in results):
        raise ValueError("Mem0 response must contain a results list of objects")
    return results


class Mem0Memory:
    """Map Gym namespaces to Mem0 user IDs without creating an SDK or provider.

    Only the SDK ``{"results": [...]}`` response envelope is supported. Provider
    exceptions propagate. Adding a record is not idempotent: metadata does not
    make a remote Mem0 write safe to retry.
    """

    def __init__(self, client: Mem0Client) -> None:
        for method in ("add", "search"):
            function = getattr(client, method, None)
            if not callable(function) or inspect.iscoroutinefunction(function):
                raise ValueError("Mem0Memory requires synchronous add/search client methods")
        self._client = client

    def add(self, record: MemoryRecord) -> None:
        """Store exact text and provenance, accepting only a committed ADD result."""
        if not isinstance(record, MemoryRecord):
            raise ValueError("record must be a MemoryRecord")
        _namespace(record.namespace)
        payload = self._client.add(
            record.text,
            user_id=record.namespace,
            metadata={
                "source_episode_id": record.source_episode_id,
                "memory_id": record.memory_id,
            },
            infer=False,
        )
        results = _results(payload)
        if len(results) != 1:
            raise ValueError("Mem0 add must acknowledge exactly one committed memory")
        result = results[0]
        _text(result.get("id"), "result id")
        if result.get("event") != "ADD" or result.get("memory") != record.text:
            raise ValueError("Mem0 add did not acknowledge the requested memory text")
        if "status" in result:
            raise ValueError("Mem0 add returned an unconfirmed memory")
        if "user_id" in result and result["user_id"] != record.namespace:
            raise ValueError("Mem0 add acknowledged a different namespace")
        if "metadata" in result and result["metadata"] != {
            "source_episode_id": record.source_episode_id,
            "memory_id": record.memory_id,
        }:
            raise ValueError("Mem0 add acknowledged different provenance")

    def search(self, query: str, *, namespace: str, limit: int = 5) -> tuple[MemoryRecord, ...]:
        """Require matching result user IDs and the original Gym provenance metadata."""
        _text(query, "query")
        _namespace(namespace)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        payload = self._client.search(query, filters={"user_id": namespace}, top_k=limit)
        records = []
        memory_ids = set()
        for item in _results(payload):
            _text(item.get("id"), "result id")
            if "status" in item or item.get("event", "ADD") not in ("ADD", "UPDATE"):
                raise ValueError("Mem0 search returned an uncommitted memory")
            if item.get("user_id") != namespace:
                raise ValueError("Mem0 search result must include the requested namespace user_id")
            metadata = item.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("Mem0 memory must include Gym provenance metadata")
            record = MemoryRecord(
                memory_id=_text(metadata.get("memory_id"), "memory_id"),
                namespace=namespace,
                text=_text(item.get("memory"), "memory"),
                source_episode_id=_text(metadata.get("source_episode_id"), "source_episode_id"),
            )
            if record.memory_id in memory_ids:
                raise ValueError("Mem0 search returned duplicate Gym memory IDs")
            memory_ids.add(record.memory_id)
            records.append(record)
        return tuple(records[:limit])

    def snapshot(
        self, queries: Iterable[str], *, namespace: str, limit: int = 5
    ) -> FrozenMem0Memory:
        """Materialize selected query results before evaluation; perform no remote writes.

        Reads are sequential, not an atomic snapshot of the remote database.
        Once captured, the returned adapter never calls the client again.
        """
        _namespace(namespace)
        if isinstance(queries, (str, bytes)) or not isinstance(queries, Iterable):
            raise ValueError("queries must be an iterable of query strings")
        query_list = tuple(queries)
        if not query_list:
            raise ValueError("snapshot requires at least one query")
        for query in query_list:
            _text(query, "query")
        values = {
            query: self.search(query, namespace=namespace, limit=limit)
            for query in dict.fromkeys(query_list)
        }
        return FrozenMem0Memory(namespace, limit, values)


@dataclass(frozen=True)
class FrozenMem0Memory:
    """Read-only, namespace-bound query results; unknown queries fail explicitly."""

    namespace: str
    limit: int
    results: Mapping[str, tuple[MemoryRecord, ...]]

    def __post_init__(self) -> None:
        _namespace(self.namespace)
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError("snapshot limit must be a positive integer")
        if not isinstance(self.results, Mapping):
            raise ValueError("snapshot results must be a query mapping")
        captured = {}
        for query, records in self.results.items():
            _text(query, "query")
            records = tuple(records)
            if len(records) > self.limit or any(
                not isinstance(record, MemoryRecord) or record.namespace != self.namespace
                for record in records
            ):
                raise ValueError("snapshot records must match namespace and limit")
            captured[query] = records
        object.__setattr__(self, "results", MappingProxyType(captured))

    def add(self, record: MemoryRecord) -> None:
        raise ValueError("Mem0 snapshot is read-only; use the live adapter for training")

    def search(self, query: str, *, namespace: str, limit: int = 5) -> tuple[MemoryRecord, ...]:
        if namespace != self.namespace:
            raise ValueError("snapshot namespace mismatch")
        if type(limit) is not int or not 1 <= limit <= self.limit:
            raise ValueError("requested limit must be within the captured snapshot limit")
        if query not in self.results:
            raise ValueError("query was not captured in this memory snapshot")
        return self.results[query][:limit]
