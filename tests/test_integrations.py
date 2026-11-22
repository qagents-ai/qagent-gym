from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest

from qagent_gym.integrations import Mem0Memory, reward_payload, send_reward
from qagent_gym.records import MemoryRecord, Reward


def memory() -> MemoryRecord:
    return MemoryRecord(
        memory_id="memory-1",
        namespace="training",
        text="Check the answer before submitting.",
        source_episode_id="episode-1",
    )


def search_result() -> dict:
    record = memory()
    return {
        "id": "provider-1",
        "memory": record.text,
        "user_id": record.namespace,
        "metadata": {
            "memory_id": record.memory_id,
            "source_episode_id": record.source_episode_id,
        },
    }


class FakeMem0:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, dict]] = []
        self.add_result: object = {
            "results": [{"id": "provider-1", "memory": memory().text, "event": "ADD"}]
        }
        self.search_result: object = {"results": [search_result()]}

    def add(self, messages: str, **kwargs: object) -> object:
        self.calls.append(("add", messages, kwargs))
        return self.add_result

    def search(self, query: str, **kwargs: object) -> object:
        self.calls.append(("search", query, kwargs))
        return self.search_result


def test_mem0_preserves_text_scope_and_provenance_without_inference() -> None:
    client = FakeMem0()
    adapter = Mem0Memory(client)
    assert adapter.add(memory()) is None
    assert client.calls == [
        (
            "add",
            memory().text,
            {
                "user_id": "training",
                "metadata": {"memory_id": "memory-1", "source_episode_id": "episode-1"},
                "infer": False,
            },
        )
    ]
    assert adapter.search("answer", namespace="training", limit=2) == (memory(),)
    assert client.calls[-1] == (
        "search",
        "answer",
        {"filters": {"user_id": "training"}, "top_k": 2},
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"event_id": "pending-1", "status": "PENDING"},
        {"results": "wrong"},
        {"results": [None]},
        {"results": []},
        {"results": [{"id": "provider-1", "memory": "different", "event": "ADD"}]},
        {"results": [{"id": "provider-1", "memory": memory().text, "event": "DELETE"}]},
        {"results": [{"id": None, "memory": memory().text, "event": "ADD"}]},
        {
            "results": [
                {"id": "provider-1", "memory": memory().text, "event": "ADD", "status": "PENDING"}
            ]
        },
    ],
)
def test_mem0_rejects_unconfirmed_or_malformed_add(payload: object) -> None:
    client = FakeMem0()
    client.add_result = payload
    with pytest.raises(ValueError):
        Mem0Memory(client).add(memory())
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"user_id": "evaluation"},
        {"metadata": None},
        {"metadata": {"memory_id": "memory-1"}},
        {"metadata": {"memory_id": 12, "source_episode_id": "episode-1"}},
        {"memory": ""},
        {"id": None},
        {"status": "PENDING"},
        {"event": "DELETE"},
        {"event": []},
    ],
)
def test_mem0_rejects_wrong_scope_or_missing_provenance(changes: dict) -> None:
    client = FakeMem0()
    item = search_result()
    item.update(changes)
    client.search_result = {"results": [item]}
    with pytest.raises(ValueError):
        Mem0Memory(client).search("answer", namespace="training")


@pytest.mark.parametrize("user_id", [None, "evaluation"])
def test_mem0_rejects_unverified_scope(user_id: str | None) -> None:
    client = FakeMem0()
    item = search_result()
    if user_id is None:
        item.pop("user_id")
    else:
        item["user_id"] = user_id
    client.search_result = {"results": [search_result(), item]}
    with pytest.raises(ValueError, match="namespace"):
        Mem0Memory(client).search("answer", namespace="training")


def test_mem0_empty_search_and_limit_are_normalized() -> None:
    client = FakeMem0()
    adapter = Mem0Memory(client)
    client.search_result = {"results": []}
    assert adapter.search("answer", namespace="training") == ()
    first, second = search_result(), deepcopy(search_result())
    second["metadata"]["memory_id"] = "memory-2"
    client.search_result = {"results": [first, second]}
    assert adapter.search("answer", namespace="training", limit=1) == (memory(),)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_mem0_rejects_invalid_limit_without_calling_client(limit: int) -> None:
    client = FakeMem0()
    with pytest.raises(ValueError):
        Mem0Memory(client).search("answer", namespace="training", limit=limit)
    assert client.calls == []


def test_lightning_builds_v1_payload_and_uses_only_supplied_transport() -> None:
    reward = Reward(value=0.5, evaluator="exact-v1", feedback="Partly correct.")
    expected = {
        "event_type": "reward",
        "data": {"value": 0.5, "source": "exact-v1", "reason": "Partly correct."},
    }
    assert reward_payload(reward) == expected
    calls = []

    def transport(url: str, **kwargs: object) -> None:
        calls.append((url, kwargs))

    send_reward(
        reward,
        event_url="http://localhost:8181/events/rollout-1",
        api_key="test-key",
        transport=transport,
    )
    assert calls == [
        (
            "http://localhost:8181/events/rollout-1",
            {"json": expected, "headers": {"Authorization": "Bearer test-key"}},
        )
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.5", 10**400])
def test_lightning_rejects_invalid_numeric_boundary(value: object) -> None:
    malformed = cast(Reward, SimpleNamespace(value=value, evaluator="test", feedback=""))
    with pytest.raises(ValueError, match="finite number"):
        reward_payload(malformed)


@pytest.mark.parametrize(
    "event_url,api_key",
    [
        ("relative/events", "test-key"),
        ("file:///events", "test-key"),
        ("https://user:password@example.test/events", "test-key"),
        ("https://example.test/has space", "test-key"),
        ("https://example.test/events", ""),
        ("https://example.test/events", "bad\nheader"),
    ],
)
def test_lightning_rejects_invalid_configuration_before_transport(
    event_url: str, api_key: str
) -> None:
    calls = []
    with pytest.raises(ValueError):
        send_reward(
            Reward(value=1, evaluator="test", feedback=""),
            event_url=event_url,
            api_key=api_key,
            transport=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


def test_lightning_transport_failure_propagates_without_retry() -> None:
    calls = []

    def failing_transport(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise TimeoutError("remote result unknown")

    with pytest.raises(TimeoutError, match="remote result unknown"):
        send_reward(
            Reward(value=1, evaluator="test", feedback=""),
            event_url="https://example.test/events",
            api_key="test-key",
            transport=failing_transport,
        )
    assert len(calls) == 1
