"""Scripted agents check the learning integration without calling a model service."""

from __future__ import annotations

import json
import re
from pathlib import Path

from qagents import Agent
from qagents.models import ModelRequest
from qagents.models.testing import FunctionModel, ScriptedModel

from .export import export_jsonl
from .gym import ExactMatchEvaluator, Gym
from .qagents_adapter import QAgentsGenerator, QAgentsReflector
from .records import Task
from .storage import SQLiteStore

LESSON = "To convert metres to centimetres, multiply the number of metres by 100."


def _answer(request: ModelRequest) -> str:
    prompt = request.messages[-1].text
    task = prompt.rsplit("TASK:\n", 1)[-1]
    metres = int(re.search(r"Convert (\d+) metres", task).group(1))
    return str(metres * 100 if LESSON in prompt else metres)


async def run_demo(
    path: str | Path, *, export_path: str | Path | None = None, namespace: str = "metres-demo"
) -> dict[str, object]:
    """Train one task and evaluate different tasks after reopening persistent storage.

    On a fresh namespace, baseline reward is 0 and post-learning rewards are 1.
    Reusing a namespace deliberately reuses its learned playbook.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = QAgentsGenerator(Agent(FunctionModel(_answer), instructions="Return one number."))
    lessons = json.dumps([{"bullet_id": "metres-to-cm", "text": LESSON, "helpful": 1}])
    reflector = QAgentsReflector(QAgentsGenerator(Agent(ScriptedModel([lessons]))))
    evaluator = ExactMatchEvaluator({"train-3m": "300", "eval-7m": "700", "eval-9m": "900"})
    with SQLiteStore(path) as store:
        gym = Gym(generator, evaluator, reflector, store)
        training = await gym.train(
            Task("train-3m", "Convert 3 metres to centimetres."), namespace=namespace
        )
    with SQLiteStore(path) as store:
        gym = Gym(generator, evaluator, reflector, store)
        before = (store.snapshot(namespace), store.memories(namespace), store.changes(namespace))
        evaluated = await gym.evaluate(
            [
                Task("eval-7m", "Convert 7 metres to centimetres.", split="eval"),
                Task("eval-9m", "Convert 9 metres to centimetres.", split="eval"),
            ],
            namespace=namespace,
        )
        after = (store.snapshot(namespace), store.memories(namespace), store.changes(namespace))
        episodes = (training, *evaluated)
        if export_path is not None:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            export_jsonl(episodes, export_path)
        return {
            "training_reward": training.reward.value,
            "evaluation_rewards": [episode.reward.value for episode in evaluated],
            "evaluation_outputs": [episode.generation.output for episode in evaluated],
            "playbook_revision": store.snapshot(namespace).revision,
            "frozen_evaluation": before == after,
            "episodes": len(episodes),
            "model": training.generation.model,
            "learning": "scripted context adaptation; no model weights updated",
        }
