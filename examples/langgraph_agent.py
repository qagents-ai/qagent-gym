"""Run a real local LangGraph through Gym and retain paused-run semantics.

Run after installing the langgraph extra: python -m examples.langgraph_agent
Nodes use deterministic Python; no LLM or network service is invoked.
"""

from __future__ import annotations

import asyncio
import json
from typing import TypedDict

from qagent_gym.gym import ExactMatchEvaluator, Gym
from qagent_gym.integrations.langgraph import LangGraphGenerator
from qagent_gym.records import Episode, Lesson, Task
from qagent_gym.storage import SQLiteStore

RULE = "Multiply unit cents by quantity before adding delivery cents."


class CartState(TypedDict):
    question: str
    context: str
    namespace: str
    task_id: str
    split: str
    output: str


class CorrectCart:
    async def reflect(self, episode: Episode) -> tuple[Lesson, ...]:
        return (Lesson("quantity", RULE),) if episode.reward.value == 0 else ()


def build_graph():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def calculate(state: CartState) -> dict:
        cart = json.loads(state["question"])
        if cart.get("needs_review"):
            interrupt({"reason": "Cart requires human review"})
        playbook = state["context"].split("\n\nRetrieved memory:", 1)[0]
        quantity = cart["quantity"] if RULE in playbook else 1
        return {"output": str(cart["unit_cents"] * quantity + cart["delivery_cents"])}

    return (
        StateGraph(CartState)
        .add_node("calculate", calculate)
        .add_edge(START, "calculate")
        .add_edge("calculate", END)
        .compile(checkpointer=InMemorySaver())
    )


async def run() -> dict:
    graph = build_graph()
    generator = LangGraphGenerator(graph)
    tasks = [
        Task("learn", json.dumps({"unit_cents": 150, "quantity": 3, "delivery_cents": 25})),
        Task(
            "heldout",
            json.dumps({"unit_cents": 220, "quantity": 2, "delivery_cents": 30}),
            split="eval",
        ),
        Task(
            "review",
            json.dumps(
                {"unit_cents": 220, "quantity": 2, "delivery_cents": 30, "needs_review": True}
            ),
            split="eval",
        ),
    ]
    with SQLiteStore() as store:
        gym = Gym(
            generator, ExactMatchEvaluator({"learn": "475", "heldout": "470"}), CorrectCart(), store
        )
        learned = await gym.train(tasks[0], namespace="cart-a")
        snapshot = store.snapshot("cart-a"), store.memories("cart-a"), store.changes("cart-a")
        final, paused = await gym.evaluate(tasks[1:], namespace="cart-a")
        threads = [
            json.loads(e.generation.steps[0].content)["thread_id"] for e in (learned, final, paused)
        ]
        return {
            "mode": "native LangGraph; deterministic nodes; no provider calls",
            "training_answer": learned.generation.output,
            "training_reward": learned.reward.value,
            "heldout_answer": final.generation.output,
            "heldout_reward": final.reward.value,
            "paused_status": paused.generation.status,
            "paused_answer": paused.generation.output,
            "distinct_checkpoint_threads": len(set(threads)),
            "frozen_evaluation": snapshot
            == (store.snapshot("cart-a"), store.memories("cart-a"), store.changes("cart-a")),
        }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), indent=2))
