"""A writing agent learns a reusable style rule from a separate reviewer agent.

Run: python -m examples.peer_feedback
Both agents use FunctionModel: grading and writing are deterministic Python,
not a claim about language-model writing quality or social consensus.
"""

from __future__ import annotations

import json
from pathlib import Path

from qagents import Agent
from qagents.models import ModelRequest
from qagents.models.testing import FunctionModel

from qagent_gym.gym import Gym
from qagent_gym.qagents_adapter import QAgentsGenerator, QAgentsReflector
from qagent_gym.records import Context, Generation, Reward, Task
from qagent_gym.storage import SQLiteStore

from ._support import has_rule, run_cli, state, task_payload, user_prompt

STYLE_RULE = "Write one concise factual sentence: name the feature and its benefit; omit hype."


def write_note(request: ModelRequest) -> str:
    item = task_payload(request)
    if has_rule(request, STYLE_RULE):
        return f"{item['feature'].capitalize()} helps teams {item['benefit']}."
    return (
        f"We are really extremely excited to announce {item['feature']}, an amazing wonderful "
        f"capability that lets our very valued users {item['benefit']}."
    )


def review_note(request: ModelRequest) -> str:
    item = task_payload(request)
    text = item["candidate"]
    problems = []
    if len(text.split()) > item["max_words"]:
        problems.append(f"Use at most {item['max_words']} words.")
    if any(word in text.casefold().split() for word in ("really", "amazing", "wonderful")):
        problems.append("Remove promotional filler.")
    if item["feature"].casefold() not in text.casefold() or item["benefit"] not in text:
        problems.append("Keep the feature and concrete benefit.")
    return json.dumps({"value": float(not problems), "feedback": " ".join(problems) or "Accepted."})


def reflect_style(request: ModelRequest) -> str:
    episode = json.loads(user_prompt(request).split("EPISODE:\n", 1)[1])
    if episode["reward"]["value"] < 1:
        return json.dumps([{"bullet_id": "concise-factual", "text": STYLE_RULE}])
    return "[]"


class PeerEvaluator:
    """Convert a separate qagents review into the Gym's typed feedback boundary."""

    def __init__(self, reviewer: QAgentsGenerator) -> None:
        self.reviewer = reviewer

    async def evaluate(self, task: Task, generation: Generation) -> Reward:
        if generation.status != "completed":
            return Reward(0, "peer-style-v1", f"Incomplete: {generation.status}")
        review = await self.reviewer.generate(
            Task(
                f"review:{task.task_id}",
                json.dumps({**json.loads(task.prompt), "candidate": generation.output}),
            ),
            Context("peer-review"),
        )
        if review.status != "completed":
            raise ValueError("peer review did not complete")
        payload = json.loads(review.output)
        return Reward(payload["value"], "peer-style-v1", payload["feedback"])


def note_task(task_id: str, feature: str, benefit: str, *, split: str = "train") -> Task:
    return Task(
        task_id, json.dumps({"feature": feature, "benefit": benefit, "max_words": 12}), split
    )


async def run(path: str | Path, *, namespace: str = "peer-writing") -> dict:
    writer = QAgentsGenerator(Agent(FunctionModel(write_note), name="Quentin-writer"))
    peer_model = FunctionModel(review_note)
    evaluator = PeerEvaluator(QAgentsGenerator(Agent(peer_model, name="Quentin-reviewer")))
    reflector = QAgentsReflector(QAgentsGenerator(Agent(FunctionModel(reflect_style))))
    with SQLiteStore(path) as store:
        gym = Gym(writer, evaluator, reflector, store)
        first = await gym.train(
            note_task("practice-1", "checkpoint recovery", "resume jobs"), namespace=namespace
        )
        second = await gym.train(
            note_task("practice-2", "tool journaling", "audit tool calls"), namespace=namespace
        )
        frozen = state(store, namespace)
        (heldout,) = await gym.evaluate(
            [note_task("heldout-1", "durable sessions", "keep useful context", split="eval")],
            namespace=namespace,
        )
        return {
            "training_rewards": [first.reward.value, second.reward.value],
            "heldout_reward": heldout.reward.value,
            "initial_text": first.generation.output,
            "peer_feedback": first.reward.feedback,
            "corrected_text": second.generation.output,
            "heldout_text": heldout.generation.output,
            "lesson_used_on_second_task": any(b.text == STYLE_RULE for b in second.context.bullets),
            "peer_reviews": len(peer_model.requests),
            "frozen_evaluation": state(store, namespace) == frozen,
        }


if __name__ == "__main__":
    run_cli(run, __doc__)
