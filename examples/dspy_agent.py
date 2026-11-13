"""Run an actual DSPy Module with Gym context learning, without an LM or optimizer.

Run after installing the dspy extra: python -m examples.dspy_agent
The module reformats structured release facts using deterministic Python.
"""

from __future__ import annotations

import asyncio
import json

from qagent_gym.gym import ExactMatchEvaluator, Gym
from qagent_gym.integrations.dspy import DSPyGenerator
from qagent_gym.records import Episode, Lesson, Task
from qagent_gym.storage import SQLiteStore

RULE = "Use feature: benefit; omit promotional adjectives."


class EditorialFeedback:
    async def reflect(self, episode: Episode) -> tuple[Lesson, ...]:
        return (Lesson("release-style", RULE),) if episode.reward.value == 0 else ()


def build_program():
    import dspy

    class ReleaseWriter(dspy.Module):
        async def aforward(self, *, question: str, context: str):
            facts = json.loads(question)
            learned = json.loads(context)["learned_context"].split("\n\nRetrieved memory:", 1)[0]
            clean = f"{facts['feature']}: {facts['benefit']}"
            return dspy.Prediction(answer=clean if RULE in learned else f"Amazing {clean}")

    return ReleaseWriter()


async def run() -> dict:
    generator = DSPyGenerator(build_program())
    training = Task("learn", json.dumps({"feature": "Export", "benefit": "Share your results"}))
    heldout = Task(
        "heldout", json.dumps({"feature": "Autosave", "benefit": "Keep your work"}), split="eval"
    )
    with SQLiteStore() as store:
        gym = Gym(
            generator,
            ExactMatchEvaluator(
                {"learn": "Export: Share your results", "heldout": "Autosave: Keep your work"}
            ),
            EditorialFeedback(),
            store,
        )
        learned = await gym.train(training, namespace="release-style")
        snapshot = store.snapshot("release-style"), store.memories("release-style")
        (final,) = await gym.evaluate([heldout], namespace="release-style")
        return {
            "mode": "native DSPy Module and Prediction; deterministic policy; no LM or optimizer",
            "training_answer": learned.generation.output,
            "training_reward": learned.reward.value,
            "heldout_answer": final.generation.output,
            "heldout_reward": final.reward.value,
            "lesson_source_run": final.context.bullets[0].source_episode_id,
            "training_run_id": learned.run_id,
            "frozen_evaluation": snapshot
            == (store.snapshot("release-style"), store.memories("release-style")),
        }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), indent=2))
