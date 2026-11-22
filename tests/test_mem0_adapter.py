import asyncio
import importlib.util
from copy import deepcopy

import pytest

from examples.mem0_memory import MemoryFixture, run
from qagent_gym.integrations.mem0 import FrozenMem0Memory, Mem0Memory
from qagent_gym.records import MemoryRecord


def record(namespace="a", memory_id="m1"):
    return MemoryRecord(memory_id, namespace, "Write concise release notes.", "training-1")


def test_snapshot_is_frozen_scoped_and_read_only_without_remote_calls():
    client = MemoryFixture()
    live = Mem0Memory(client)
    live.add(record())
    live.add(record("b", "other"))
    snapshot = live.snapshot(["release", "release"], namespace="a")
    reads = client.search_calls
    client.rows.clear()
    assert snapshot.search("release", namespace="a") == (record(),)
    assert client.search_calls == reads
    with pytest.raises(ValueError, match="read-only"):
        snapshot.add(record())
    with pytest.raises(ValueError, match="namespace"):
        snapshot.search("release", namespace="b")
    with pytest.raises(ValueError, match="not captured"):
        snapshot.search("new question", namespace="a")
    with pytest.raises(TypeError):
        snapshot.results["other"] = ()


@pytest.mark.parametrize("queries", [[], "release", ["valid", ""], [None], None])
def test_snapshot_rejects_invalid_queries_before_remote_reads(queries):
    client = MemoryFixture()
    with pytest.raises(ValueError):
        Mem0Memory(client).snapshot(queries, namespace="a")
    assert client.search_calls == 0


@pytest.mark.parametrize("namespace", [" a", "a ", "\ta"])
def test_namespace_cannot_be_silently_trimmed_by_sdk(namespace):
    client = MemoryFixture()
    memory = Mem0Memory(client)
    with pytest.raises(ValueError, match="whitespace"):
        memory.add(record(namespace))
    with pytest.raises(ValueError, match="whitespace"):
        memory.search("release", namespace=namespace)
    assert client.add_calls == client.search_calls == 0


def test_conflicting_duplicate_memory_ids_fail_closed():
    client = MemoryFixture()
    live = Mem0Memory(client)
    live.add(record())
    other = deepcopy(client.rows[0])
    other["memory"] = "Contradictory instructions."
    client.rows.append(other)
    with pytest.raises(ValueError, match="duplicate"):
        live.search("release", namespace="a")


@pytest.mark.parametrize(
    "mismatch",
    [
        {"user_id": "b"},
        {"metadata": {"memory_id": "wrong", "source_episode_id": "training-1"}},
    ],
)
def test_add_scope_or_provenance_mismatch_is_not_acknowledged(mismatch):
    class Bad(MemoryFixture):
        def add(self, *args, **kwargs):
            result = super().add(*args, **kwargs)
            result["results"][0].update(mismatch)
            return result

    with pytest.raises(ValueError):
        Mem0Memory(Bad()).add(record())


def test_async_client_is_explicitly_rejected():
    class AsyncClient(MemoryFixture):
        async def search(self, *args, **kwargs):
            return {}

    with pytest.raises(ValueError, match="synchronous"):
        Mem0Memory(AsyncClient())


def test_unknown_remote_write_outcome_propagates_without_retry():
    class Interrupted(MemoryFixture):
        def add(self, *args, **kwargs):
            super().add(*args, **kwargs)
            raise TimeoutError("acknowledgement lost")

    client = Interrupted()
    with pytest.raises(TimeoutError, match="acknowledgement lost"):
        Mem0Memory(client).add(record())
    assert client.add_calls == 1
    assert len(client.rows) == 1


def test_snapshot_constructor_copies_inputs_and_validates_scope():
    inputs = {"release": [record()]}
    snapshot = FrozenMem0Memory("a", 5, inputs)
    inputs["release"].clear()
    assert snapshot.search("release", namespace="a") == (record(),)
    with pytest.raises(ValueError, match="namespace"):
        FrozenMem0Memory("a", 5, {"release": [record("b")]})
    with pytest.raises(ValueError, match="limit"):
        snapshot.search("release", namespace="a", limit=6)


def test_mem0_example_executes_real_qagents_and_freezes_external_memory():
    report = asyncio.run(run())
    assert report["training_reward"] == report["evaluation_reward"] == 1
    assert report["answer"] == "Η έκδοση είναι έτοιμη."
    assert report["captured_namespace"] == "workspace-a"
    assert report["captured_source_episode"] == report["training_run_id"]
    assert report["evaluation_remote_reads"] == report["evaluation_remote_writes"] == 0
    assert not report["other_namespace_leaked"]
    assert report["agent_calls"] == 2


@pytest.mark.skipif(importlib.util.find_spec("mem0") is None, reason="optional native Mem0 SDK")
def test_native_mem0_memory_add_search_and_snapshot_with_local_qdrant(tmp_path, monkeypatch):
    """Real Memory + local Qdrant; replace only inference providers with deterministic fixtures."""
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0-config"))
    monkeypatch.setenv("MEM0_TELEMETRY", "false")
    from mem0 import Memory
    from mem0.utils.factory import EmbedderFactory, LlmFactory
    from qdrant_client import QdrantClient

    class NoInference:
        def generate_response(self, *args, **kwargs):
            raise AssertionError("infer=False must not call an LLM")

    class LocalEmbedding:
        def embed(self, text, *args, **kwargs):
            return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(LlmFactory, "create", lambda *args, **kwargs: NoInference())
    monkeypatch.setattr(EmbedderFactory, "create", lambda *args, **kwargs: LocalEmbedding())
    qdrant = QdrantClient(location=":memory:")
    sdk = Memory.from_config(
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "client": qdrant,
                    "path": str(tmp_path / "unused-qdrant"),
                    "collection_name": "gym_native",
                    "embedding_model_dims": 4,
                },
            },
            "embedder": {"provider": "openai", "config": {"embedding_dims": 4}},
            "history_db_path": str(tmp_path / "history.sqlite3"),
        }
    )
    try:
        memory = Mem0Memory(sdk)
        memory.add(record())
        memory.add(record("b", "other"))
        matches = memory.search("release notes", namespace="a")
        assert matches == (record(),)
        frozen = memory.snapshot(["release notes"], namespace="a")
        assert frozen.search("release notes", namespace="a") == matches
    finally:
        sdk.close()
        qdrant.close()
