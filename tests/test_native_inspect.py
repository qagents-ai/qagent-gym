"""Optional native-reader contract; run in an environment with inspect-ai installed."""

import socket

import pytest

from qagent_gym.integrations import write_inspect_dataset
from qagent_gym.records import Task


def test_export_loads_with_native_inspect_reader(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("native dataset loading must not contact a service")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    inspect_dataset = pytest.importorskip("inspect_ai.dataset")
    tasks = [
        Task("test-unicode", "Πόσο κάνει 2 + 2;", "eval"),
        Task("test-empty", "Return an empty string.", "eval"),
    ]
    path = tmp_path / "inspect.jsonl"
    write_inspect_dataset(
        tasks,
        {"test-unicode": "4", "test-empty": ""},
        path,
        excluded_task_ids=["training-task", "validation-task"],
    )
    samples = list(inspect_dataset.json_dataset(str(path)))
    assert [(sample.id, sample.input, sample.target) for sample in samples] == [
        ("test-unicode", tasks[0].prompt, "4"),
        ("test-empty", tasks[1].prompt, ""),
    ]
    assert all(
        sample.metadata == {"qagent_gym_split": "eval", "partition": "test"} for sample in samples
    )
