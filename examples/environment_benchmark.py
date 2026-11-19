"""Replay the same Gymnasium tasks through QAgents and native LangGraph.

Install both optional extras. No model provider is contacted.
python -m examples.environment_benchmark --output-dir .gym/environment-benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import TypedDict

from qagents import Agent, Limits, RunResult, TrialTarget, compare_reliability, reliability
from qagents.evals import Case, Dataset, ExactMatch, evaluate
from qagents.models.testing import FunctionModel

from qagent_gym.integrations.gymnasium import GymnasiumEnvironment
from qagent_gym.integrations.langgraph import LangGraphGenerator
from qagent_gym.qagents_adapter import QAgentsGenerator
from qagent_gym.records import Context, Task
from qagent_gym.rollouts import GeneratorPolicy, run_episode

ROUTE = (1, 1, 2, 1, 2, 2)


def choose(prompt: str) -> str:
    history = json.loads(prompt.rsplit("Previous observations and actions (JSON):\n", 1)[1])
    return str(ROUTE[len(history) % len(ROUTE)])


class State(TypedDict):
    question: str
    context: str
    namespace: str
    task_id: str
    split: str
    output: str


def make_policy(framework, task, context):
    if framework == "qagents":

        def decision(request):
            prompt = next(
                message.text for message in reversed(request.messages) if message.role == "user"
            )
            return choose(prompt)

        generator = QAgentsGenerator(
            Agent(FunctionModel(decision)),
            limits=Limits(max_turns=2, max_total_tokens=2048, max_cost_usd=0.05, timeout_s=5),
        )
    else:
        from langgraph.graph import END, START, StateGraph

        def decision(state: State):
            return {"output": choose(state["question"])}

        graph = (
            StateGraph(State)
            .add_node("decide", decision)
            .add_edge(START, "decide")
            .add_edge("decide", END)
            .compile()
        )
        generator = LangGraphGenerator(graph)
    return GeneratorPolicy(generator, task, context)


async def run(*, seeds=tuple(range(20)), output_dir: Path | None = None) -> dict:
    import gymnasium

    seeds = tuple(seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    context = Context("environment-contract")
    cases = Dataset(
        [
            Case(
                name="deterministic",
                prompt="Reach FrozenLake's goal with discrete actions.",
                expected="1",
                metadata={"is_slippery": False},
            ),
            Case(
                name="slippery",
                prompt="Reach FrozenLake's goal with discrete actions.",
                expected="1",
                metadata={"is_slippery": True},
            ),
        ],
        name="FrozenLake-boundary-contract",
    )
    revision = hashlib.sha256(
        json.dumps(
            {
                "seeds": seeds,
                "max_episode_steps": 12,
                "route": ROUTE,
                "cases": [case.model_dump(mode="json", exclude={"deps"}) for case in cases.cases],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    trajectories, reports = {}, {}
    for framework in ("qagents", "langgraph"):
        trajectories[framework] = {}

        async def trial(case, repetition, framework=framework):
            seed = seeds[repetition]
            task = Task(case.name, case.prompt, split="eval")
            environment = GymnasiumEnvironment(
                gymnasium.make(
                    "FrozenLake-v1",
                    is_slippery=case.metadata["is_slippery"],
                    max_episode_steps=12,
                )
            )
            rollout = await run_episode(
                environment,
                make_policy(framework, task, context),
                task,
                context,
                seed=seed,
                max_steps=14,
                timeout_s=30,
            )
            trajectories[framework][(case.name, seed)] = rollout
            if output_dir is not None:
                rollout.save(output_dir / f"{framework}-{case.name}-{seed}.json")
            completed = rollout.terminated
            return RunResult(
                run_id=rollout.run_id,
                session_id=f"rollout:{rollout.run_id}",
                agent=framework,
                status="completed" if completed else "limit_reached",
                output=str(int(rollout.total_reward == 1)) if completed else None,
                stop_reason=rollout.stop_reason,
            )

        reports[framework] = await evaluate(
            TrialTarget(trial),
            cases,
            [ExactMatch()],
            repeat=len(seeds),
            concurrency=4,
            name=framework,
            metadata={"dataset_revision": revision, "seeds": list(seeds)},
        )
        if reports[framework].summary.errors:
            raise AssertionError(
                f"{framework} integration failed: {reports[framework].summary.errors}"
            )

    def signature(rollout):
        return (
            rollout.initial_observation,
            rollout.stop_reason,
            tuple(
                (step.observation, step.decision.output, step.result)
                for step in rollout.transitions
            ),
        )

    agreement = all(
        signature(rollout) == signature(trajectories["langgraph"][key])
        for key, rollout in trajectories["qagents"].items()
    )
    assert agreement
    assert len(trajectories["qagents"]) == len(cases) * len(seeds)
    report = {
        "mode": "native environments and frameworks; scripted decisions; no provider calls",
        "seeds": list(seeds),
        "cases_per_framework": len(cases),
        "rollouts": sum(len(items) for items in trajectories.values()),
        "trajectory_agreement": agreement,
        "dataset_revision": revision,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "gymnasium",
                "langgraph",
                "langchain-core",
                "pydantic",
            )
        },
        "platform": platform.platform(),
        "reliability": {
            name: reliability(value, ks=(1, 3) if len(seeds) >= 3 else (1,)).model_dump(mode="json")
            for name, value in reports.items()
        },
        "paired": compare_reliability(reports["qagents"], reports["langgraph"]).model_dump(
            mode="json"
        ),
        "endings": {
            name: {
                reason: sum(r.stop_reason == reason for r in items.values())
                for reason in ("terminated", "truncated", "max_steps", "policy_stopped")
            }
            for name, items in trajectories.items()
        },
        "usage_capture": "not collected by this boundary benchmark",
    }
    if output_dir is not None:
        with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(output_dir=args.output_dir)), indent=2))
