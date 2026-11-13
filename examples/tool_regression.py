"""Learn from a cart-total bug, then regression-test real qagents tool calls.

Run: python -m examples.tool_regression
The offline policy initially drops quantities. Feedback changes its tool
arguments, while the arithmetic tool itself stays correct and unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from qagents import Agent, ToolCallPart, tool
from qagents.models import ModelRequest
from qagents.models.testing import FunctionModel

from qagent_gym.gym import ExactMatchEvaluator, Gym
from qagent_gym.qagents_adapter import QAgentsGenerator
from qagent_gym.records import Episode, Lesson, Task
from qagent_gym.storage import SQLiteStore

from ._support import has_rule, run_cli, state, task_payload

QUANTITY_RULE = "Pass each requested quantity to line_total; sum every returned line total."


@tool(read_only=True)
def line_total(unit_cents: int, quantity: int) -> int:
    """Return the price of one cart line in integer cents."""
    if unit_cents < 0 or quantity < 0:
        raise ValueError("price and quantity must be nonnegative")
    return unit_cents * quantity


def cart_policy(request: ModelRequest):
    if request.messages[-1].role == "tool":
        return str(
            sum(int(part.content) for message in request.messages for part in message.tool_results)
        )
    item = task_payload(request)
    use_quantities = has_rule(request, QUANTITY_RULE)
    return [
        ToolCallPart(
            id=f"line-{index}",
            name="line_total",
            arguments={
                "unit_cents": line["unit_cents"],
                "quantity": line["quantity"] if use_quantities else 1,
            },
        )
        for index, line in enumerate(item["lines"])
    ]


class QuantityReflector:
    """An explicit task-specific reflector; replace it with QAgentsReflector for a model."""

    async def reflect(self, episode: Episode) -> tuple[Lesson, ...]:
        if episode.reward.value < 1:
            return (Lesson("respect-quantity", QUANTITY_RULE),)
        return ()


def cart_task(task_id: str, lines: list[tuple[int, int]], *, split: str = "train") -> Task:
    return Task(
        task_id,
        json.dumps(
            {"lines": [{"unit_cents": price, "quantity": quantity} for price, quantity in lines]}
        ),
        split,
    )


async def run(path: str | Path, *, namespace: str = "cart-regression") -> dict:
    training = cart_task("practice-cart", [(125, 3), (200, 2)])
    heldout = [
        cart_task("mixed-quantities", [(150, 4), (75, 2)], split="eval"),
        cart_task("single-items", [(99, 1), (201, 1)], split="eval"),
    ]
    answers = {
        task.task_id: str(
            sum(line["unit_cents"] * line["quantity"] for line in json.loads(task.prompt)["lines"])
        )
        for task in [training, *heldout]
    }
    generator = QAgentsGenerator(Agent(FunctionModel(cart_policy), tools=[line_total]))
    with SQLiteStore(path) as store:
        gym = Gym(generator, ExactMatchEvaluator(answers), QuantityReflector(), store)
        learned = await gym.train(training, namespace=namespace)
        frozen = state(store, namespace)
        results = await gym.evaluate(heldout, namespace=namespace)
        return {
            "training_reward": learned.reward.value,
            "training_output_cents": learned.generation.output,
            "training_expected_cents": answers[training.task_id],
            "evaluation_rewards": [episode.reward.value for episode in results],
            "evaluation_outputs_cents": [episode.generation.output for episode in results],
            "tool_trace": [
                {"kind": step.kind, "name": step.name, "content": step.content}
                for step in results[0].generation.steps
                if step.kind != "message"
            ],
            "frozen_evaluation": state(store, namespace) == frozen,
        }


if __name__ == "__main__":
    run_cli(run, __doc__)
