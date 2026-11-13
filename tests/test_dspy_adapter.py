import asyncio
import importlib.util
import json

import pytest

from qagent_gym.integrations.dspy import DSPyGenerator
from qagent_gym.records import Bullet, Context, Task


class FakePrediction:
    def __init__(self, values):
        self.values = values

    def toDict(self):
        return self.values


class FakeProgram:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_program_called_with_explicit_context_and_normalized_output():
    program = FakeProgram(FakePrediction({"answer": "ready", "unused": object()}))
    context = Context("workspace-a", bullets=(Bullet("style", "Keep it short", "run-1"),))
    result = asyncio.run(DSPyGenerator(program).generate(Task("t", "draft", split="eval"), context))
    assert result.output == "ready" and result.status == "completed"
    assert program.calls[0]["question"] == "draft"
    payload = json.loads(program.calls[0]["context"])
    assert payload["namespace"] == "workspace-a" and payload["split"] == "eval"
    assert "Keep it short" in payload["learned_context"]
    assert len(result.steps) == 3
    assert result.steps[-1].content == "ready"


@pytest.mark.parametrize(
    "result",
    [
        None,
        "answer",
        {},
        FakePrediction([]),
        FakePrediction({}),
        FakePrediction({"answer": 1}),
        FakePrediction({"answer": " "}),
    ],
)
def test_invalid_prediction_does_not_become_a_completed_episode(result):
    with pytest.raises(ValueError):
        asyncio.run(DSPyGenerator(FakeProgram(result)).generate(Task("t", "q"), Context("n")))


def test_program_exception_propagates_without_retry():
    calls = []

    class Broken:
        async def acall(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("module failed")

    with pytest.raises(RuntimeError, match="module failed"):
        asyncio.run(DSPyGenerator(Broken()).generate(Task("t", "q"), Context("n")))
    assert len(calls) == 1


def test_context_limit_rejects_before_call():
    program = FakeProgram(FakePrediction({"answer": "ready"}))
    with pytest.raises(ValueError, match="context exceeds"):
        asyncio.run(
            DSPyGenerator(program, max_context_chars=1).generate(Task("t", "q"), Context("n"))
        )
    assert program.calls == []


@pytest.mark.parametrize(
    "kwargs", [{"output_field": ""}, {"max_context_chars": True}, {"max_context_chars": 0}]
)
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        DSPyGenerator(FakeProgram(None), **kwargs)


@pytest.mark.skipif(importlib.util.find_spec("dspy") is None, reason="optional native DSPy")
def test_native_dspy_module_prediction_and_context_learning(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_CACHEDIR", str(tmp_path / "dspy-cache"))
    from examples.dspy_agent import run

    report = asyncio.run(run())
    assert report["training_answer"] == "Amazing Export: Share your results"
    assert report["training_reward"] == 0
    assert report["heldout_answer"] == "Autosave: Keep your work"
    assert report["heldout_reward"] == 1
    assert report["lesson_source_run"] == report["training_run_id"]
    assert report["frozen_evaluation"]
