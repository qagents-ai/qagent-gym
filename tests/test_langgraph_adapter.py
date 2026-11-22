import asyncio
import importlib.util
import json
from types import SimpleNamespace

import pytest

from qagent_gym.integrations.langgraph import LangGraphGenerator
from qagent_gym.records import Bullet, Context, Task


class FakeGraph:
    def __init__(self, result=None, *, checkpointed=False, next_nodes=()):
        self.checkpointer = object() if checkpointed else None
        self.result = result or SimpleNamespace(value={"output": "answer"}, interrupts=())
        self.next_nodes = next_nodes
        self.calls = []

    async def ainvoke(self, inputs, config, **kwargs):
        self.calls.append((inputs, config, kwargs))
        return self.result

    async def aget_state(self, config):
        assert config == self.calls[-1][1]
        return SimpleNamespace(next=self.next_nodes)


def test_context_and_scope_are_injected_with_fresh_bounded_threads():
    graph = FakeGraph()
    generator = LangGraphGenerator(graph, recursion_limit=8)
    task = Task("same-id", "calculate", split="eval")
    first = Context("a", bullets=(Bullet("rule", "Check quantity", "source"),))
    for context in (first, first, Context("b")):
        answer = asyncio.run(generator.generate(task, context))
        assert answer.output == "answer" and answer.status == "completed"
    threads = [config["configurable"]["thread_id"] for _, config, _ in graph.calls]
    assert len(set(threads)) == 3
    assert all(len(thread) < 255 for thread in threads)
    assert "Check quantity" in graph.calls[0][0]["context"]
    assert graph.calls[0][0]["namespace"] == "a"
    assert graph.calls[2][0]["namespace"] == "b"
    assert graph.calls[0][1]["metadata"]["gym_split"] == "eval"
    assert graph.calls[0][1]["recursion_limit"] == 8
    assert graph.calls[0][2] == {"version": "v2", "stream_mode": "values"}
    assert json.loads(answer.steps[0].content)["thread_id"] == threads[-1]


@pytest.mark.parametrize("dynamic", [True, False])
def test_dynamic_and_static_pauses_never_expose_partial_answers(dynamic):
    graph = FakeGraph(
        SimpleNamespace(
            value={"output": "partial answer"}, interrupts=("approval",) if dynamic else ()
        ),
        checkpointed=True,
        next_nodes=("pending",),
    )
    result = asyncio.run(LangGraphGenerator(graph).generate(Task("t", "question"), Context("n")))
    assert result.status == "paused" and result.output == ""
    assert not any(step.role == "assistant" for step in result.steps)


@pytest.mark.parametrize(
    "result",
    [
        {},
        None,
        SimpleNamespace(value={"output": "answer"}, interrupts=None),
        SimpleNamespace(value="answer", interrupts=()),
        SimpleNamespace(value={"output": 3}, interrupts=()),
        SimpleNamespace(value={"output": " "}, interrupts=()),
    ],
)
def test_rejects_bad_outputs_instead_of_recording_completion(result):
    graph = FakeGraph()
    graph.result = result
    with pytest.raises(ValueError):
        asyncio.run(LangGraphGenerator(graph).generate(Task("t", "question"), Context("n")))


def test_provider_errors_propagate_without_retry():
    class Broken(FakeGraph):
        async def ainvoke(self, *args, **kwargs):
            self.calls.append("failed")
            raise TimeoutError("unknown remote outcome")

    graph = Broken()
    with pytest.raises(TimeoutError):
        asyncio.run(LangGraphGenerator(graph).generate(Task("t", "question"), Context("n")))
    assert graph.calls == ["failed"]


def test_context_limit_checked_before_graph_execution():
    graph = FakeGraph()
    with pytest.raises(ValueError, match="context exceeds"):
        asyncio.run(
            LangGraphGenerator(graph, max_context_chars=1).generate(Task("t", "q"), Context("n"))
        )
    assert not graph.calls


def test_paused_training_records_no_learning_or_memory():
    from qagent_gym.gym import ExactMatchEvaluator, Gym
    from qagent_gym.storage import SQLiteStore

    class MustNotReflect:
        async def reflect(self, episode):
            raise AssertionError("paused generations must not train")

    graph = FakeGraph(SimpleNamespace(value={"output": "partial"}, interrupts=("approval",)))
    with SQLiteStore() as store:
        gym = Gym(LangGraphGenerator(graph), ExactMatchEvaluator({}), MustNotReflect(), store)
        episode = asyncio.run(gym.train(Task("t", "question"), namespace="n"))
        assert episode.generation.status == "paused" and episode.reward.value == 0
        assert store.snapshot("n").revision == 0
        assert store.memories("n") == ()


@pytest.mark.parametrize(
    "kwargs", [{"recursion_limit": True}, {"output_key": ""}, {"max_context_chars": 0}]
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        LangGraphGenerator(FakeGraph(), **kwargs)


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None, reason="optional native LangGraph"
)
def test_native_graph_learning_and_dynamic_pause(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    from examples.langgraph_agent import run

    report = asyncio.run(run())
    assert report["training_answer"] == "175" and report["training_reward"] == 0
    assert report["heldout_answer"] == "470" and report["heldout_reward"] == 1
    assert report["paused_status"] == "paused" and report["paused_answer"] == ""
    assert report["distinct_checkpoint_threads"] == 3
    assert report["frozen_evaluation"]


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None, reason="optional native LangGraph"
)
def test_native_static_interrupt_after_answer_is_incomplete(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from examples.langgraph_agent import CartState

    graph = (
        StateGraph(CartState)
        .add_node("draft", lambda _: {"output": "not final"})
        .add_node("finish", lambda _: {"output": "final"})
        .add_edge(START, "draft")
        .add_edge("draft", "finish")
        .add_edge("finish", END)
        .compile(checkpointer=InMemorySaver(), interrupt_after=["draft"])
    )
    result = asyncio.run(LangGraphGenerator(graph).generate(Task("t", "question"), Context("n")))
    assert result.status == "paused" and result.output == ""
