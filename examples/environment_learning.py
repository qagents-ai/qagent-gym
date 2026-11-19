"""Learn a review procedure through a stateful environment and frozen evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from qagents import Agent, Limits
from qagents.models.testing import FunctionModel, ScriptedModel

from qagent_gym.gym import Gym
from qagent_gym.qagents_adapter import QAgentsGenerator, QAgentsReflector
from qagent_gym.records import Task
from qagent_gym.rollouts import (
    EnvironmentResult,
    EnvironmentReturnEvaluator,
    GeneratorPolicy,
    Rollout,
    RolloutGenerator,
)
from qagent_gym.storage import SQLiteStore

from ._support import has_rule, run_cli, state, user_prompt

RULE_ID = "review-before-submit"


class ReviewEnvironment:
    """A local checklist fixture: review, apply the correction, submit."""

    async def reset(self, *, seed):
        self.stage = "draft"
        return json.dumps({"stage": self.stage, "actions": ["review", "revise", "submit"]})

    async def step(self, action):
        terminated, reward = False, 0.0
        if action == "review" and self.stage == "draft":
            self.stage = "reviewed"
        elif action == "revise" and self.stage == "reviewed":
            self.stage = "revised"
        elif action == "submit":
            reward = float(self.stage == "revised")
            self.stage, terminated = "submitted", True
        else:
            raise ValueError("action is invalid for this review stage")
        return EnvironmentResult(
            json.dumps({"stage": self.stage}),
            reward,
            terminated=terminated,
            info_json='{"verifier":"ordered-checklist-v1"}',
        )

    async def close(self):
        pass


def policy(task, context):
    def choose(request):
        if not has_rule(request, RULE_ID):
            return "submit"
        observation = (
            user_prompt(request)
            .split("Current observation:\n", 1)[1]
            .split("\n\nPrevious observations", 1)[0]
        )
        return {"draft": "review", "reviewed": "revise", "revised": "submit"}[
            json.loads(observation)["stage"]
        ]

    generator = QAgentsGenerator(
        Agent(FunctionModel(choose)),
        limits=Limits(max_turns=2, max_total_tokens=2048, max_cost_usd=0.05, timeout_s=5),
    )
    return GeneratorPolicy(generator, task, context)


async def run(path: Path, *, namespace="review-environment") -> dict:
    reviewer = QAgentsReflector(
        QAgentsGenerator(
            Agent(
                ScriptedModel(
                    [
                        json.dumps(
                            [
                                {
                                    "bullet_id": RULE_ID,
                                    "text": ("Review, revise the correction, then submit."),
                                }
                            ]
                        )
                    ]
                )
            )
        )
    )
    with SQLiteStore(path) as store:
        gym = Gym(
            RolloutGenerator(lambda task: ReviewEnvironment(), policy, max_steps=4),
            EnvironmentReturnEvaluator(),
            reviewer,
            store,
        )
        training = await gym.train(
            Task("practice-review", "Review the sample brief and submit the corrected draft."),
            namespace=namespace,
        )
        before = state(store, namespace)
        evaluated = await gym.evaluate(
            [
                Task(
                    "heldout-review-a",
                    "Review the second brief and submit its correction.",
                    split="eval",
                ),
                Task(
                    "heldout-review-b",
                    "Review the third brief and submit its correction.",
                    split="eval",
                ),
            ],
            namespace=namespace,
        )
        assert state(store, namespace) == before
        rollouts = [Rollout.from_json(item.generation.steps[0].content) for item in evaluated]
        return {
            "training_reward": training.reward.value,
            "heldout_rewards": [item.reward.value for item in evaluated],
            "heldout_actions": [
                [step.decision.output for step in rollout.transitions] for rollout in rollouts
            ],
            "frozen_evaluation": True,
            "playbook_revision": store.snapshot(namespace).revision,
        }


if __name__ == "__main__":
    run_cli(run, __doc__)
