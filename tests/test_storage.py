import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import pytest

from qagent_gym.records import (
    Context,
    Episode,
    Generation,
    MemoryRecord,
    PlaybookDelta,
    Reward,
    Task,
)
from qagent_gym.storage import SQLiteStore


def delta(delta_id="d1", bullet_id="b1", **kwargs):
    return PlaybookDelta(
        delta_id, bullet_id, "Check units before calculating.", "episode-1", **kwargs
    )


def test_persistence_scope_search_and_immutable_episode(tmp_path):
    path = tmp_path / "gym.sqlite3"
    episode = Episode(
        "run-1",
        Task("t1", "Check units"),
        Generation("42"),
        Reward(1.0, "exact-v1", "Correct"),
        Context("alpha"),
    )
    with SQLiteStore(path) as store:
        store.add(MemoryRecord("m1", "alpha", "Check units carefully", "run-1"))
        store.add(MemoryRecord("m2", "beta", "Check units elsewhere", "run-2"))
        assert store.apply("alpha", [delta()]) == 1
        store.save_episode(episode)
    with SQLiteStore(path) as store:
        assert [m.memory_id for m in store.search("units", namespace="alpha")] == ["m1"]
        assert store.snapshot("alpha").revision == 1
        assert store.snapshot("beta").bullets == ()
        assert store.episodes("alpha") == (episode,)
        assert Episode.from_json(episode.to_json()) == episode
        store.delete("m1", namespace="alpha")
        store.delete("m1", namespace="alpha")
        assert store.memories("alpha") == ()
        assert len(store.memories("beta")) == 1
    with pytest.raises(FrozenInstanceError):
        episode.reward.value = 2


def test_delta_replay_conflict_and_transaction_rollback():
    with SQLiteStore() as store:
        change = delta()
        assert store.apply("a", [change, change]) == 1
        assert store.apply("a", [change]) == 0
        with pytest.raises(ValueError, match="different payload"):
            store.apply("a", [replace(change, text="Changed")])
        with pytest.raises(ValueError, match="unknown bullet"):
            store.apply("a", [delta("d2", "b2"), delta("d3", "missing", operation="update")])
        assert store.snapshot("a").revision == 1
        assert len(store.changes("a")) == 1
        assert store.apply("a", [delta("d4", operation="update", helpful=2)]) == 1
        assert store.snapshot("a").bullets[0].helpful == 2
        store.apply("a", [delta("d5", operation="deprecate", harmful=1)])
        assert not store.snapshot("a").bullets[0].active
        assert "Check units" not in store.snapshot("a").render()


def test_duplicate_concurrent_deltas(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    with SQLiteStore(path):
        pass

    def apply_once(_):
        with SQLiteStore(path) as store:
            return store.apply("a", [delta()])

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(apply_once, range(12))) == 1
    with SQLiteStore(path) as store:
        assert store.snapshot("a").revision == 1


def test_memory_and_episode_ids_cannot_change_payload():
    with SQLiteStore() as store:
        memory = MemoryRecord("m1", "a", "A fact", "r1")
        store.add(memory)
        store.add(memory)
        with pytest.raises(ValueError, match="memory_id"):
            store.add(replace(memory, text="Another fact"))
        episode = Episode(
            "r1",
            Task("t1", "Question"),
            Generation("Answer"),
            Reward(0, "e1", "Feedback"),
            Context("a"),
        )
        store.save_episode(episode)
        store.save_episode(episode)
        with pytest.raises(ValueError, match="run_id"):
            store.save_episode(replace(episode, reward=Reward(1, "e1", "Changed")))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10**400, True, "1"])
def test_bad_rewards(value):
    with pytest.raises(ValueError):
        Reward(value, "e1", "Feedback")


@pytest.mark.parametrize("payload", ["{}", "null", "[]", "not json"])
def test_bad_episode_json(payload):
    with pytest.raises(ValueError, match="invalid episode"):
        Episode.from_json(payload)


def test_json_import_rejects_unrepresentable_reward():
    episode = Episode(
        "r", Task("t", "Question"), Generation("Answer"), Reward(1, "e", "Feedback"), Context("a")
    )
    data = json.loads(episode.to_json())
    data["reward"]["value"] = 10**400
    with pytest.raises(ValueError, match="finite"):
        Episode.from_json(json.dumps(data))


def test_boundary_validation():
    with pytest.raises(ValueError):
        delta(operation="replace_all")
    with pytest.raises(ValueError):
        delta(helpful=-1)
    with pytest.raises(ValueError):
        Context("a", memories=[MemoryRecord("m", "b", "text", "r")])
    with pytest.raises(ValueError):
        Generation("An answer", status="blocked")
