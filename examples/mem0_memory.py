"""Train a workspace preference and evaluate qagents against a frozen Mem0 snapshot.

Run: python -m examples.mem0_memory
The default client is an in-process contract fixture, not a live Mem0 service.
"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy

from qagents import Agent, Limits
from qagents.models import ModelRequest
from qagents.models.testing import FunctionModel

from qagent_gym.gym import ExactMatchEvaluator, Gym
from qagent_gym.integrations.mem0 import Mem0Memory
from qagent_gym.qagents_adapter import QAgentsGenerator
from qagent_gym.records import Episode, Lesson, MemoryRecord, Task
from qagent_gym.storage import SQLiteStore

from ._support import task_payload, user_prompt


class MemoryFixture:
    """Synchronous SDK contract fixture; retrieval is namespace-only, not semantic."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.add_calls = 0
        self.search_calls = 0

    def add(self, messages, *, user_id, metadata, infer):
        if infer:
            raise ValueError("this example stores explicit text")
        self.add_calls += 1
        row = {
            "id": f"fixture-{self.add_calls}",
            "memory": messages,
            "user_id": user_id,
            "metadata": deepcopy(metadata),
        }
        self.rows.append(row)
        return {"results": [{"id": row["id"], "event": "ADD", "memory": messages}]}

    def search(self, query, *, filters, top_k):
        self.search_calls += 1
        return {
            "results": deepcopy(
                [row for row in self.rows if row["user_id"] == filters["user_id"]][:top_k]
            )
        }


class NoLessons:
    async def reflect(self, episode: Episode) -> tuple[Lesson, ...]:
        return ()


async def run() -> dict:
    client = MemoryFixture()
    memory = Mem0Memory(client)
    memory.add(MemoryRecord("other", "workspace-b", "Respond in English.", "seed-other"))

    def policy(request: ModelRequest) -> str:
        payload = task_payload(request)
        if payload["action"] == "remember":
            return payload["preference"]
        remembered = (
            user_prompt(request).split("Retrieved memory:\n", 1)[1].split("\n\nTASK:", 1)[0]
        )
        return "Η έκδοση είναι έτοιμη." if "Respond in Greek." in remembered else "Release ready."

    model = FunctionModel(policy)
    generator = QAgentsGenerator(Agent(model), limits=Limits(max_turns=2, timeout_s=10))
    training = Task(
        "preference", json.dumps({"action": "remember", "preference": "Respond in Greek."})
    )
    heldout = Task("release", json.dumps({"action": "release-note"}), split="eval")
    evaluator = ExactMatchEvaluator(
        {"preference": "Respond in Greek.", "release": "Η έκδοση είναι έτοιμη."}
    )
    with SQLiteStore() as store:
        gym = Gym(generator, evaluator, NoLessons(), store, memory=memory)
        learned = await gym.train(training, namespace="workspace-a")
        frozen = memory.snapshot([heldout.prompt], namespace="workspace-a")
        captured = frozen.search(heldout.prompt, namespace="workspace-a")
        # Simulate external writes after capture. Evaluation must not read this new value.
        client.rows = [row for row in client.rows if row["user_id"] != "workspace-a"]
        memory.add(MemoryRecord("changed", "workspace-a", "Respond in English.", "external-change"))
        writes_before, reads_before = client.add_calls, client.search_calls
        evaluation = Gym(generator, evaluator, NoLessons(), store, memory=frozen)
        (final,) = await evaluation.evaluate([heldout], namespace="workspace-a")
        return {
            "mode": "real qagents execution; in-process Mem0 contract fixture",
            "training_reward": learned.reward.value,
            "evaluation_reward": final.reward.value,
            "answer": final.generation.output,
            "captured_namespace": captured[0].namespace,
            "captured_source_episode": captured[0].source_episode_id,
            "training_run_id": learned.run_id,
            "evaluation_remote_reads": client.search_calls - reads_before,
            "evaluation_remote_writes": client.add_calls - writes_before,
            "other_namespace_leaked": any(record.namespace != "workspace-a" for record in captured),
            "agent_calls": len(model.requests),
        }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2))
