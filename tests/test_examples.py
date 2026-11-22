import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples import (
    before_after,
    evaluation_exports,
    integration_contracts,
    peer_feedback,
    tool_regression,
)
from qagent_gym.storage import SQLiteStore


def test_peer_feedback_is_reused_on_different_writing_tasks(tmp_path):
    path = tmp_path / "peer.sqlite3"
    report = asyncio.run(peer_feedback.run(path))
    assert report["training_rewards"] == [0, 1]
    assert report["heldout_reward"] == 1
    assert report["peer_reviews"] == 3
    assert report["lesson_used_on_second_task"] is True
    assert report["frozen_evaluation"] is True
    assert "Remove promotional filler" in report["peer_feedback"]
    assert "Tool journaling" in report["corrected_text"]
    assert "Durable sessions" in report["heldout_text"]
    assert len(report["heldout_text"].split()) <= 12
    with SQLiteStore(path) as store:
        episodes = store.episodes("peer-writing")
        assert len(store.changes("peer-writing")) == 1
        assert episodes[1].context.bullets[0].source_episode_id == episodes[0].run_id
        assert episodes[2].task.split == "eval"
        assert {memory.source_episode_id for memory in store.memories("peer-writing")} == {
            episodes[0].run_id,
            episodes[1].run_id,
        }


def test_cart_policy_changes_tool_arguments_and_keeps_regression_case(tmp_path):
    path = tmp_path / "tools.sqlite3"
    report = asyncio.run(tool_regression.run(path))
    assert report["training_reward"] == 0
    assert report["training_output_cents"] == "325"
    assert report["training_expected_cents"] == "775"
    assert report["evaluation_rewards"] == [1, 1]
    assert report["evaluation_outputs_cents"] == ["750", "300"]
    assert report["frozen_evaluation"] is True
    calls = [
        json.loads(step["content"]) for step in report["tool_trace"] if step["kind"] == "tool_call"
    ]
    assert calls == [{"unit_cents": 150, "quantity": 4}, {"unit_cents": 75, "quantity": 2}]
    assert sorted(
        int(step["content"]) for step in report["tool_trace"] if step["kind"] == "tool_result"
    ) == [150, 600]
    with SQLiteStore(path) as store:
        first = store.episodes("cart-regression")[0]
        assert all(
            json.loads(step.content)["quantity"] == 1
            for step in first.generation.steps
            if step.kind == "tool_call"
        )


def test_fixed_split_comparison_is_reproducible_and_does_not_leak_eval_memory(tmp_path):
    path = tmp_path / "comparison.sqlite3"
    first = asyncio.run(before_after.run(path))
    second = asyncio.run(before_after.run(tmp_path / "replicate.sqlite3"))
    assert first == second
    assert first["score_before"] == 0.25
    assert first["score_after"] == 1
    assert set(first["train_task_ids"]).isdisjoint(first["eval_task_ids"])
    assert first["baseline_frozen"] is first["post_training_frozen"] is True
    assert first["only_training_memory"] is True
    assert [row["after"] for row in first["rows"]] == ["420", "240", "60", "0"]
    with SQLiteStore(path) as store:
        episodes = store.episodes("frozen-comparison")
        assert len(episodes) == 10
        assert [episode.context.revision for episode in episodes[:4]] == [0] * 4
        assert [episode.context.revision for episode in episodes[-4:]] == [2] * 4
        training_ids = {episode.run_id for episode in episodes if episode.task.split == "train"}
        assert all(
            memory.source_episode_id in training_ids
            for episode in episodes
            for memory in episode.context.memories
        )
    with pytest.raises(ValueError, match="empty namespace"):
        asyncio.run(before_after.run(path))


def test_fake_integrations_preserve_scope_and_provenance():
    report = integration_contracts.run()
    assert report["memory_matches"] == [
        {
            "memory_id": "rule-1",
            "namespace": "project-a",
            "text": "Verify the output units.",
            "source_episode_id": "training-1",
        }
    ]
    assert report["transport_calls"] == 1
    assert report["reward_event"] == {
        "event_type": "reward",
        "data": {"value": 1, "source": "offline-check-v1", "reason": "Expected result matched."},
    }


def test_evaluation_exports_keep_heldout_truth_separate_from_reflection(tmp_path):
    path = tmp_path / "exports.sqlite3"
    report = asyncio.run(evaluation_exports.run(path))
    assert report["inspect_samples"] == 4
    assert report["gepa_training_examples"] == 2
    rows = [json.loads(line) for line in Path(report["inspect_path"]).read_text().splitlines()]
    assert [row["target"] for row in rows] == ["420", "240", "60", "0"]
    assert all(row["metadata"]["partition"] == "test" for row in rows)
    reflection = json.loads(Path(report["gepa_path"]).read_text())["playbook"]
    with SQLiteStore(path) as store:
        episodes = store.episodes("evaluation-exports")
        training_ids = {episode.run_id for episode in episodes if episode.task.split == "train"}
        heldout_ids = {episode.run_id for episode in episodes if episode.task.split == "eval"}
        assert {row["episode_id"] for row in reflection} == training_ids
        assert not heldout_ids.intersection(row["episode_id"] for row in reflection)
    with pytest.raises(ValueError, match="already exist"):
        asyncio.run(evaluation_exports.run(path))


@pytest.mark.parametrize(
    "module",
    [
        "peer_feedback",
        "tool_regression",
        "before_after",
        "integration_contracts",
        "evaluation_exports",
    ],
)
def test_example_module_entrypoints(module, tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", f"examples.{module}"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    report = json.loads(result.stdout)
    assert "mode" in report
    assert result.stderr == ""
