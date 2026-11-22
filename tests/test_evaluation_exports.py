import json
from dataclasses import replace

import pytest

from qagent_gym.integrations.evaluation import gepa_reflective_dataset, write_inspect_dataset
from qagent_gym.records import Context, Episode, Generation, Reward, Task


def episode(run_id="run-1"):
    return Episode(
        run_id,
        Task("practice-1", "Convert three metres to centimetres."),
        Generation("3", model="test:scripted"),
        Reward(0.0, "exact-v1", "Multiply metres by 100."),
        Context("experiment-a"),
    )


def test_inspect_jsonl_keeps_separate_truth_and_unicode(tmp_path):
    path = tmp_path / "test.jsonl"
    tasks = [Task("heldout-1", "Πόσο κάνει 2 + 2;", "eval"), Task("heldout-2", "Empty?", "eval")]
    assert (
        write_inspect_dataset(
            iter(tasks),
            {"heldout-1": "4", "heldout-2": ""},
            path,
            excluded_task_ids=["practice-1", "validation-1"],
        )
        == 2
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "id": "heldout-1",
            "input": tasks[0].prompt,
            "target": "4",
            "metadata": {"qagent_gym_split": "eval", "partition": "test"},
        },
        {
            "id": "heldout-2",
            "input": "Empty?",
            "target": "",
            "metadata": {"qagent_gym_split": "eval", "partition": "test"},
        },
    ]
    assert tasks[0].prompt == "Πόσο κάνει 2 + 2;"
    with pytest.raises(FileExistsError):
        write_inspect_dataset(tasks, {"heldout-1": "4", "heldout-2": ""}, path)


@pytest.mark.parametrize(
    "tasks,targets,options",
    [
        ([], {}, {}),
        ([Task("a", "Q")], {"a": "A"}, {}),
        ([Task("a", "Q", "eval")] * 2, {"a": "A"}, {}),
        ([Task("a", "Q", "eval")], {}, {}),
        ([Task("a", "Q", "eval")], {"a": "A", "extra": "B"}, {}),
        ([Task("a", "Q", "eval")], {"a": 1}, {}),
        ([Task("a", "Q", "eval")], {"a": "A"}, {"excluded_task_ids": ["a"]}),
        ([Task("a", "Q", "eval")], {"a": "A"}, {"excluded_task_ids": [None]}),
        ([Task("a", "Q", "eval")], {"a": "A"}, {"excluded_task_ids": [["a"]]}),
        ([Task("task-a", "Q", "eval")], {"task-a": "A"}, {"excluded_task_ids": "task-a"}),
        ([Task("a", "Q", "eval")], {"a": "A"}, {"partition": "train"}),
    ],
)
def test_inspect_invalid_data_writes_no_partial_file(tmp_path, tasks, targets, options):
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError):
        write_inspect_dataset(tasks, targets, path, **options)
    assert not path.exists()


def test_inspect_labels_optimizer_validation_explicitly(tmp_path):
    path = tmp_path / "validation.jsonl"
    write_inspect_dataset([Task("v", "Q", "eval")], {"v": "A"}, path, partition="validation")
    assert json.loads(path.read_text())["metadata"]["partition"] == "validation"


@pytest.mark.parametrize("field", ["input", "target", "id"])
def test_invalid_unicode_in_later_row_leaves_no_export(tmp_path, field):
    invalid = chr(0xD800)
    task_id = invalid if field == "id" else "bad"
    tasks = [
        Task("good", "Good input", "eval"),
        Task(task_id, invalid if field == "input" else "Question", "eval"),
    ]
    targets = {"good": "Answer", task_id: invalid if field == "target" else "Answer"}
    path = tmp_path / "surrogate.jsonl"
    with pytest.raises(ValueError):
        write_inspect_dataset(tasks, targets, path)
    assert not path.exists()


def test_gepa_reflections_preserve_alignment_and_provenance():
    first = episode()
    second = replace(
        episode("run-2"),
        generation=Generation("300"),
        reward=Reward(1.0, "exact-v1", "Correct."),
        context=Context("experiment-a", revision=1),
    )
    data = gepa_reflective_dataset(iter([first, second]), component="system_prompt")
    rows = data["system_prompt"]
    assert [row["episode_id"] for row in rows] == ["run-1", "run-2"]
    assert [row["score"] for row in rows] == [0.0, 1.0]
    assert [row["Generated Outputs"] for row in rows] == ["3", "300"]
    assert rows[0]["Inputs"] == {
        "task": first.task.prompt,
        "learned_context": first.context.render(),
    }
    assert rows[0]["Feedback"] == "Multiply metres by 100."
    assert rows[1]["context_revision"] == 1
    assert json.loads(json.dumps(data, allow_nan=False)) == data


@pytest.mark.parametrize(
    "episodes,component",
    [
        ([], "playbook"),
        ([episode()], ""),
        ([episode(), episode()], "playbook"),
        ([replace(episode(), task=Task("test", "Q", "eval"))], "playbook"),
        ([replace(episode(), generation=Generation("", status="paused"))], "playbook"),
        ([episode(), replace(episode("run-2"), reward=Reward(1, "other-scale", "F"))], "playbook"),
        (["not an episode"], "playbook"),
    ],
)
def test_gepa_rejects_leakage_incomplete_and_misaligned_data(episodes, component):
    with pytest.raises(ValueError):
        gepa_reflective_dataset(episodes, component=component)
