import asyncio
import json

import pytest
from qagents import Agent, Limits, ToolCallPart, tool
from qagents.models.testing import ScriptedModel

from qagent_gym.cli import main
from qagent_gym.demo import run_demo
from qagent_gym.export import export_jsonl, read_jsonl
from qagent_gym.gym import DeltaCurator, ExactMatchEvaluator, Gym
from qagent_gym.qagents_adapter import QAgentsGenerator, QAgentsReflector
from qagent_gym.records import Context, Episode, Generation, Lesson, Reward, Task
from qagent_gym.storage import SQLiteStore


class NoReflection:
    async def reflect(self, episode):
        raise AssertionError("Evaluation must not invoke the reflector")


def test_real_qagents_demo_reopens_state_and_freezes_eval(tmp_path):
    path = tmp_path / "gym.sqlite3"
    exported = tmp_path / "episodes.jsonl"
    report = asyncio.run(run_demo(path, export_path=exported))
    assert report["training_reward"] == 0
    assert report["evaluation_rewards"] == [1, 1]
    assert report["evaluation_outputs"] == ["700", "900"]
    assert report["frozen_evaluation"] is True
    assert report["model"] == "test:function"
    episodes = read_jsonl(exported)
    assert episodes[0].context.revision == 0
    assert episodes[1].context.revision == 1
    assert episodes[1].context.bullets[0].source_episode_id == episodes[0].run_id
    assert "Expected '700'" not in episodes[1].generation.steps[0].content
    with SQLiteStore(path) as store:
        assert store.episodes("metres-demo") == episodes
        assert len(store.memories("metres-demo")) == 1
    # Existing namespaces retain learned rules; the CLI does not reset user memory.
    assert asyncio.run(run_demo(path))["training_reward"] == 1


def test_eval_snapshot_and_external_peer_feedback(tmp_path):
    seen = []

    class Generator:
        async def generate(self, task, context):
            seen.append(context)
            return Generation("candidate")

    class PeerEvaluator:
        async def evaluate(self, task, generation):
            return Reward(0.5, "peer-review-v1", "Cite a source next time.")

    with SQLiteStore(tmp_path / "gym.sqlite3") as store:
        gym = Gym(Generator(), PeerEvaluator(), NoReflection(), store)
        result = asyncio.run(
            gym.evaluate([Task("a", "Q1", "eval"), Task("b", "Q2", "eval")], namespace="peer")
        )
        assert [episode.reward.value for episode in result] == [0.5, 0.5]
        assert seen == [Context("peer"), Context("peer")]
        assert store.memories("peer") == store.changes("peer") == ()
        assert store.snapshot("peer").revision == 0
        with pytest.raises(ValueError, match="training requires"):
            asyncio.run(gym.train(Task("a", "Q1", "eval"), namespace="peer"))
        with pytest.raises(ValueError, match="evaluation requires"):
            asyncio.run(gym.evaluate([Task("a", "Q1")], namespace="peer"))


def test_incomplete_result_not_adapted():
    class Incomplete:
        async def generate(self, task, context):
            return Generation("", status="paused")

    with SQLiteStore() as store:
        gym = Gym(Incomplete(), ExactMatchEvaluator({}), NoReflection(), store)
        episode = asyncio.run(gym.train(Task("a", "Q1"), namespace="a"))
        assert episode.reward.value == 0
        assert store.memories("a") == ()


@pytest.mark.parametrize(
    "reply",
    [
        '{"text": "no array"}',
        '[{"text":"missing id"}]',
        '[{"bullet_id":"x","text":"bad","helpful":-1}]',
    ],
)
def test_invalid_reflection_keeps_durable_episode_but_no_learning(reply):
    with SQLiteStore() as store:
        generator = QAgentsGenerator(Agent(ScriptedModel(["wrong"])))
        reflector = QAgentsReflector(QAgentsGenerator(Agent(ScriptedModel([reply]))))
        gym = Gym(generator, ExactMatchEvaluator({"t": "right"}), reflector, store)
        with pytest.raises(ValueError, match="invalid reflection"):
            asyncio.run(gym.train(Task("t", "Question"), namespace="a"))
        assert len(store.episodes("a")) == 1
        assert store.memories("a") == store.changes("a") == ()


def test_curator_rejects_duplicate_lessons():
    episode = Episode("r", Task("t", "Q"), Generation("A"), Reward(0, "e", "F"), Context("a"))
    with pytest.raises(ValueError, match="duplicate"):
        DeltaCurator().curate(episode, (Lesson("b", "rule 1"), Lesson("b", "rule 2")))


def test_jsonl_refuses_overwrite_and_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path)
    with pytest.raises(FileExistsError):
        export_jsonl([], path)


def test_adapter_budgets_and_context_boundary():
    with pytest.raises(ValueError, match="max_turns"):
        QAgentsGenerator(Agent(ScriptedModel([])), limits=Limits(max_turns=None, timeout_s=1))
    generator = QAgentsGenerator(Agent(ScriptedModel([])), max_context_chars=1)
    with pytest.raises(ValueError, match="context exceeds"):
        asyncio.run(generator.generate(Task("t", "Q"), Context("a")))


def test_qagents_observable_tool_trace_and_paused_status():
    @tool(read_only=True)
    def double(number: int) -> int:
        """Double a number."""
        return number * 2

    model = ScriptedModel([ToolCallPart(id="call-1", name="double", arguments={"number": 2}), "4"])
    result = asyncio.run(
        QAgentsGenerator(Agent(model, tools=[double])).generate(Task("t", "Double 2"), Context("a"))
    )
    assert result.output == "4"
    assert [(s.kind, s.name, s.content) for s in result.steps if s.kind != "message"] == [
        ("tool_call", "double", '{"number": 2}'),
        ("tool_result", "double", "4"),
    ]

    @tool(requires_approval=True)
    def change() -> str:
        """Perform a change requiring approval."""
        raise AssertionError("Unapproved tool must not execute")

    paused_model = ScriptedModel([ToolCallPart(id="approval-1", name="change", arguments={})])
    paused = asyncio.run(
        QAgentsGenerator(Agent(paused_model, tools=[change])).generate(
            Task("t2", "Make a change"), Context("a")
        )
    )
    assert paused.status == "paused"
    assert paused.output == ""


def test_heldout_answer_key_stays_in_evaluator():
    model = ScriptedModel(["candidate"])
    with SQLiteStore() as store:
        gym = Gym(
            QAgentsGenerator(Agent(model)),
            ExactMatchEvaluator({"heldout": "SECRET_EXPECTED_ANSWER"}),
            NoReflection(),
            store,
        )
        asyncio.run(gym.evaluate([Task("heldout", "Question", "eval")], namespace="a"))
    assert all(
        "SECRET_EXPECTED_ANSWER" not in message.text
        for request in model.requests
        for message in request.messages
    )


def test_cli_demo_and_export(tmp_path, capsys):
    db = str(tmp_path / "demo.sqlite3")
    assert main(["demo", "--db", db]) == 0
    assert json.loads(capsys.readouterr().out)["frozen_evaluation"] is True
    target = str(tmp_path / "out.jsonl")
    assert main(["export", "--db", db, "--namespace", "metres-demo", "--output", target]) == 0
    assert json.loads(capsys.readouterr().out)["exported_episodes"] == 3
