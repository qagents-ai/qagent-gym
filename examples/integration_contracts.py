"""Exercise Mem0 and Lightning boundaries using in-process fakes, with no service.

Run: python -m examples.integration_contracts
Replace the injected clients/transports with explicitly configured real ones only
when ready to contact those services. This recipe does not perform RL training.
"""

from __future__ import annotations

import json

from qagent_gym.integrations import Mem0Memory, send_reward
from qagent_gym.records import MemoryRecord, Reward


class FakeMem0:
    """A small fake for the documented v2 response contract, not a Mem0 emulator."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, messages: str, *, user_id: str, metadata: dict, infer: bool) -> dict:
        if infer:
            raise ValueError("this recipe stores explicit memories")
        row = {
            "id": f"fake-{len(self.rows)}",
            "memory": messages,
            "user_id": user_id,
            "metadata": dict(metadata),
        }
        self.rows.append(row)
        return {"results": [{"id": row["id"], "memory": messages, "event": "ADD"}]}

    def search(self, query: str, *, filters: dict, top_k: int) -> dict:
        return {
            "results": [
                dict(row)
                for row in self.rows
                if row["user_id"] == filters["user_id"]
                and query.casefold() in row["memory"].casefold()
            ][:top_k]
        }


def run() -> dict:
    memory = Mem0Memory(FakeMem0())
    memory.add(MemoryRecord("rule-1", "project-a", "Verify the output units.", "training-1"))
    memory.add(MemoryRecord("rule-2", "project-b", "Verify a different project.", "training-2"))
    matches = memory.search("verify", namespace="project-a")
    sent = []

    def fake_transport(url: str, *, json: dict, headers: dict) -> None:
        # A real transport must validate the response and raise on failure.
        # This fake records payloads only; it does not resolve or contact the URL.
        sent.append({"url": url, "payload": json, "authenticated": "Authorization" in headers})

    send_reward(
        Reward(1, "offline-check-v1", "Expected result matched."),
        event_url="https://example.invalid/reward",
        api_key="offline-example-only",
        transport=fake_transport,
    )
    return {
        "mode": "in-process fake clients; no live integration or weight training",
        "memory_matches": [
            {
                "memory_id": match.memory_id,
                "namespace": match.namespace,
                "text": match.text,
                "source_episode_id": match.source_episode_id,
            }
            for match in matches
        ],
        "reward_event": sent[0]["payload"],
        "transport_calls": len(sent),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
