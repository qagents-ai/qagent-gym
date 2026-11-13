"""Mem0 v2 response normalization using an explicitly configured client."""

from collections.abc import Mapping
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


def _results(payload: object) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping) or "status" in payload:
        raise ValueError("Mem0 must return committed results, not a pending acknowledgement")
    results = payload.get("results")
    if not isinstance(results, list) or any(not isinstance(item, Mapping) for item in results):
        raise ValueError("Mem0 response must contain a results list of objects")
    return results


class Mem0Memory:
    """Map Gym namespaces to Mem0 user IDs without creating an SDK or provider.

    Only the v2 ``{"results": [...]}`` response envelope is supported. Provider
    exceptions propagate. Adding a record is not idempotent: metadata does not
    make a remote Mem0 write safe to retry.
    """

    def __init__(self, client: Mem0Client) -> None:
        self._client = client

    def add(self, record: MemoryRecord) -> None:
        """Store exact text and provenance, accepting only a committed ADD result."""
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

    def search(self, query: str, *, namespace: str, limit: int = 5) -> tuple[MemoryRecord, ...]:
        """Require matching result user IDs and the original Gym provenance metadata."""
        _text(query, "query")
        _text(namespace, "namespace")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        payload = self._client.search(query, filters={"user_id": namespace}, top_k=limit)
        records = []
        for item in _results(payload):
            _text(item.get("id"), "result id")
            if "status" in item or item.get("event", "ADD") not in ("ADD", "UPDATE"):
                raise ValueError("Mem0 search returned an uncommitted memory")
            if item.get("user_id") != namespace:
                raise ValueError("Mem0 search result must include the requested namespace user_id")
            metadata = item.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("Mem0 memory must include Gym provenance metadata")
            records.append(
                MemoryRecord(
                    memory_id=_text(metadata.get("memory_id"), "memory_id"),
                    namespace=namespace,
                    text=_text(item.get("memory"), "memory"),
                    source_episode_id=_text(metadata.get("source_episode_id"), "source_episode_id"),
                )
            )
        return tuple(records[:limit])
