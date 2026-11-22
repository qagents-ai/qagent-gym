"""Compare identical held-out tasks before/after training with frozen contexts.

Run: python -m examples.before_after
Requires an empty namespace. Scores describe this deterministic scripted policy,
not a benchmark claim about a pretrained model.
"""

from __future__ import annotations

import json
from pathlib import Path

from qagents import Agent
from qagents.models import ModelRequest
from qagents.models.testing import FunctionModel

from qagent_gym.gym import ExactMatchEvaluator, Gym
from qagent_gym.qagents_adapter import QAgentsGenerator, QAgentsReflector
from qagent_gym.records import Task
from qagent_gym.storage import SQLiteStore

from ._support import has_rule, run_cli, state, task_payload, user_prompt

RULES = {
    "minutes-to-seconds": "To convert minutes to seconds, multiply the number of minutes by 60.",
    "hours-to-minutes": "To convert hours to minutes, multiply the number of hours by 60.",
}


def conversion_policy(request: ModelRequest) -> str:
    item = task_payload(request)
    factor = 60 if has_rule(request, RULES[item["conversion"]]) else 1
    return str(item["value"] * factor)


def reflect_conversion(request: ModelRequest) -> str:
    episode = json.loads(user_prompt(request).split("EPISODE:\n", 1)[1])
    if episode["reward"]["value"] == 1:
        return "[]"
    conversion = json.loads(episode["task"]["prompt"])["conversion"]
    return json.dumps([{"bullet_id": conversion, "text": RULES[conversion]}])


def conversion_task(task_id: str, conversion: str, value: int, *, split: str) -> Task:
    return Task(task_id, json.dumps({"conversion": conversion, "value": value}), split)


async def run(path: str | Path, *, namespace: str = "frozen-comparison") -> dict:
    training = [
        conversion_task("train-minutes", "minutes-to-seconds", 2, split="train"),
        conversion_task("train-hours", "hours-to-minutes", 3, split="train"),
    ]
    heldout = [
        conversion_task("eval-minutes-7", "minutes-to-seconds", 7, split="eval"),
        conversion_task("eval-hours-4", "hours-to-minutes", 4, split="eval"),
        conversion_task("eval-minutes-1", "minutes-to-seconds", 1, split="eval"),
        conversion_task("eval-hours-0", "hours-to-minutes", 0, split="eval"),
    ]
    answers = {
        task.task_id: str(json.loads(task.prompt)["value"] * 60) for task in [*training, *heldout]
    }
    generator = QAgentsGenerator(Agent(FunctionModel(conversion_policy)))
    reflector = QAgentsReflector(QAgentsGenerator(Agent(FunctionModel(reflect_conversion))))
    with SQLiteStore(path) as store:
        if (
            store.snapshot(namespace).revision
            or store.memories(namespace)
            or store.episodes(namespace)
        ):
            raise ValueError("before/after comparison requires an empty namespace")
        gym = Gym(generator, ExactMatchEvaluator(answers), reflector, store)
        initial_state = state(store, namespace)
        before = await gym.evaluate(heldout, namespace=namespace)
        baseline_frozen = initial_state == state(store, namespace)
        trained = [await gym.train(task, namespace=namespace) for task in training]
        learned_state = state(store, namespace)
        after = await gym.evaluate(heldout, namespace=namespace)
        training_ids = {episode.run_id for episode in trained}
        return {
            "train_task_ids": [task.task_id for task in training],
            "eval_task_ids": [task.task_id for task in heldout],
            "training_rewards": [episode.reward.value for episode in trained],
            "score_before": sum(episode.reward.value for episode in before) / len(before),
            "score_after": sum(episode.reward.value for episode in after) / len(after),
            "baseline_frozen": baseline_frozen,
            "post_training_frozen": learned_state == state(store, namespace),
            "only_training_memory": all(
                memory.source_episode_id in training_ids for memory in store.memories(namespace)
            ),
            "rows": [
                {
                    "task_id": old.task.task_id,
                    "expected": answers[old.task.task_id],
                    "before": old.generation.output,
                    "after": new.generation.output,
                    "before_reward": old.reward.value,
                    "after_reward": new.reward.value,
                }
                for old, new in zip(before, after, strict=True)
            ],
        }


if __name__ == "__main__":
    run_cli(run, __doc__)
