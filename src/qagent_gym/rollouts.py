"""Bounded, observable interaction with a caller-supplied environment.

Environment observations and actions are text. Rewards and end-state flags come
from the environment, separately from the policy's generation.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from .gym import Generator
from .prompts import ENVIRONMENT_ACTION_PROMPT
from .records import Bullet, Context, Generation, MemoryRecord, Reward, Step, Task, _text


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(text: str):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


@dataclass(frozen=True)
class EnvironmentResult:
    """One environment response; info_json is diagnostic data, hidden from the policy."""

    observation: str
    reward: float
    terminated: bool = False
    truncated: bool = False
    info_json: str = "{}"

    def __post_init__(self) -> None:
        _text(self.observation, "observation", empty=True)
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise ValueError("environment reward must be numeric")
        try:
            finite = math.isfinite(self.reward)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("environment reward must be finite")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("termination flags must be booleans")
        if not isinstance(self.info_json, str) or not isinstance(_loads(self.info_json), dict):
            raise ValueError("info_json must encode a JSON object")


@dataclass(frozen=True)
class Transition:
    """An immutable observation, completed policy decision and environment response."""

    observation: str
    decision: Generation
    result: EnvironmentResult

    def __post_init__(self) -> None:
        _text(self.observation, "observation", empty=True)
        if not isinstance(self.decision, Generation) or self.decision.status != "completed":
            raise ValueError("transitions require a completed policy decision")
        _text(self.decision.output, "action")
        if not isinstance(self.result, EnvironmentResult):
            raise ValueError("result must be EnvironmentResult")


@dataclass(frozen=True)
class Rollout:
    """Versioned trajectory. Termination ends a task, but does not imply success."""

    run_id: str
    task: Task
    context: Context
    seed: int
    initial_observation: str
    transitions: tuple[Transition, ...]
    stop_reason: Literal["terminated", "truncated", "max_steps", "policy_stopped"]
    policy_stop: Generation | None = None
    max_steps: int = 20
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        if not isinstance(self.task, Task) or not isinstance(self.context, Context):
            raise ValueError("rollout requires Task and Context")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported rollout schema_version")
        _text(self.initial_observation, "initial_observation", empty=True)
        values = tuple(self.transitions)
        object.__setattr__(self, "transitions", values)
        if type(self.max_steps) is not int or self.max_steps < 1 or len(values) > self.max_steps:
            raise ValueError("rollout exceeds its positive max_steps budget")
        if self.stop_reason == "max_steps" and len(values) != self.max_steps:
            raise ValueError("max_steps stop requires reaching the recorded step budget")
        observation = self.initial_observation
        for index, value in enumerate(values):
            if not isinstance(value, Transition) or value.observation != observation:
                raise ValueError("rollout observations must form a continuous chain")
            if index < len(values) - 1 and (value.result.terminated or value.result.truncated):
                raise ValueError("a rollout cannot step after an environment ended")
            observation = value.result.observation
        ended = values[-1].result if values else None
        if self.stop_reason == "terminated":
            valid = ended is not None and ended.terminated
        elif self.stop_reason == "truncated":
            valid = ended is not None and ended.truncated and not ended.terminated
        elif self.stop_reason in ("max_steps", "policy_stopped"):
            valid = ended is None or not (ended.terminated or ended.truncated)
        else:
            valid = False
        if not valid:
            raise ValueError("stop_reason disagrees with environment flags")
        if self.stop_reason == "policy_stopped":
            if (
                not isinstance(self.policy_stop, Generation)
                or self.policy_stop.status == "completed"
            ):
                raise ValueError("policy_stopped requires an incomplete generation")
        elif self.policy_stop is not None:
            raise ValueError("policy_stop only belongs to policy_stopped rollouts")
        if self.stop_reason != "policy_stopped" and not values:
            raise ValueError("a rollout needs at least one transition")
        try:
            finite = math.isfinite(self.total_reward)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("total reward must be finite")

    @property
    def total_reward(self) -> float:
        return math.fsum(item.result.reward for item in self.transitions)

    @property
    def terminated(self) -> bool:
        return bool(self.transitions and self.transitions[-1].result.terminated)

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_steps" or bool(
            self.transitions and self.transitions[-1].result.truncated
        )

    def to_json(self) -> str:
        return json.dumps(
            {"format": "qagent-gym-rollout", **asdict(self)},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text: str) -> Rollout:
        """Reject unknown schemas, inconsistent chains and ambiguous JSON."""
        try:
            data = _loads(text)
            if data.pop("format") != "qagent-gym-rollout":
                raise ValueError("unsupported rollout format")
            if set(data) != set(cls.__dataclass_fields__):
                raise ValueError("unknown or missing rollout fields")
            context = data["context"]
            context["bullets"] = tuple(Bullet(**value) for value in context["bullets"])
            context["memories"] = tuple(MemoryRecord(**value) for value in context["memories"])

            def generation(value):
                value["steps"] = tuple(Step(**step) for step in value["steps"])
                return Generation(**value)

            data["transitions"] = tuple(
                Transition(
                    **{
                        **item,
                        "decision": generation(item["decision"]),
                        "result": EnvironmentResult(**item["result"]),
                    },
                )
                for item in data["transitions"]
            )
            if data.get("policy_stop") is not None:
                data["policy_stop"] = generation(data["policy_stop"])
            data["task"] = Task(**data["task"])
            data["context"] = Context(**context)
            return cls(**data)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
            raise ValueError(f"invalid rollout JSON: {error}") from error

    def save(self, path: str | Path) -> None:
        """Write a new UTF-8 trajectory file; refuse to replace another run."""
        encoded = self.to_json()
        with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")


class Environment(Protocol):
    async def reset(self, *, seed: int) -> str: ...
    async def step(self, action: str) -> EnvironmentResult: ...
    async def close(self) -> None: ...


class Policy(Protocol):
    async def act(self, observation: str, history: tuple[Transition, ...]) -> Generation: ...


def _validate_limits(seed: int, max_steps: int, timeout_s: float) -> None:
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError("timeout_s must be finite and positive")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")


async def run_episode(
    environment: Environment,
    policy: Policy,
    task: Task,
    context: Context,
    *,
    seed: int = 0,
    max_steps: int = 20,
    timeout_s: float = 60,
) -> Rollout:
    """Reset, interact, and always close the environment, including on failure.

    Timeout and cancellation propagate. Deadlines are cooperative; a native
    environment call must return before its serialized cleanup can finish.
    Each call owns its environment and policy. No automatic restart is attempted.
    """
    try:
        if not isinstance(task, Task) or not isinstance(context, Context):
            raise ValueError("run_episode requires Task and Context")
        _validate_limits(seed, max_steps, timeout_s)
        async with asyncio.timeout(timeout_s):
            observation = await environment.reset(seed=seed)
            _text(observation, "observation", empty=True)
            initial, history = observation, []
            stop, incomplete = "max_steps", None
            for _ in range(max_steps):
                decision = await policy.act(observation, tuple(history))
                if not isinstance(decision, Generation):
                    raise ValueError("policy must return Generation")
                if decision.status != "completed":
                    stop, incomplete = "policy_stopped", decision
                    break
                _text(decision.output, "action")
                result = await environment.step(decision.output)
                history.append(Transition(observation, decision, result))
                observation = result.observation
                if result.terminated or result.truncated:
                    stop = "terminated" if result.terminated else "truncated"
                    break
            return Rollout(
                uuid.uuid4().hex,
                task,
                context,
                seed,
                initial,
                tuple(history),
                stop,
                incomplete,
                max_steps=max_steps,
            )
    finally:
        await environment.close()


class GeneratorPolicy:
    """Use a Gym generator for each decision, with a fixed task and context.

    Compatible with QAgentsGenerator, LangGraphGenerator and other generators.
    Their execution budgets apply to each decision; max_steps and the rollout
    deadline bound the whole interaction. Diagnostic info is never sent to the
    policy. The visible history contains only observations and chosen actions.
    """

    def __init__(
        self, generator: Generator, task: Task, context: Context, *, max_prompt_chars: int = 20000
    ) -> None:
        if type(max_prompt_chars) is not int or max_prompt_chars < 1:
            raise ValueError("max_prompt_chars must be a positive integer")
        self.generator, self.task, self.context = generator, task, context
        self.max_prompt_chars = max_prompt_chars

    async def act(self, observation: str, history: tuple[Transition, ...]) -> Generation:
        visible = [
            {"observation": item.observation, "action": item.decision.output} for item in history
        ]
        prompt = ENVIRONMENT_ACTION_PROMPT.format(
            task=self.task.prompt,
            observation=observation,
            history=json.dumps(visible, ensure_ascii=False),
        )
        if len(prompt) > self.max_prompt_chars:
            raise ValueError("environment policy prompt exceeds max_prompt_chars")
        return await self.generator.generate(
            Task(f"{self.task.task_id}:step-{len(history)}", prompt, self.task.split),
            self.context,
        )


class RolloutGenerator:
    """Run a fresh environment and policy for each Gym task.

    The serialized trajectory is supplied by this runner, not by a model. Pair
    this generator with EnvironmentReturnEvaluator. Each factory owns one trial;
    serialize training per namespace as with other Gym generators.
    """

    def __init__(
        self,
        environment_factory: Callable[[Task], Environment],
        policy_factory: Callable[[Task, Context], Policy],
        *,
        seed: int = 0,
        max_steps: int = 20,
        timeout_s: float = 60,
    ) -> None:
        _validate_limits(seed, max_steps, timeout_s)
        if not callable(environment_factory) or not callable(policy_factory):
            raise ValueError("environment and policy factories must be callable")
        self.environment_factory = environment_factory
        self.policy_factory = policy_factory
        self.seed, self.max_steps, self.timeout_s = seed, max_steps, timeout_s

    async def generate(self, task: Task, context: Context) -> Generation:
        environment = self.environment_factory(task)
        try:
            policy = self.policy_factory(task, context)
        except BaseException:
            await environment.close()
            raise
        rollout = await run_episode(
            environment,
            policy,
            task,
            context,
            seed=self.seed,
            max_steps=self.max_steps,
            timeout_s=self.timeout_s,
        )
        encoded = rollout.to_json()
        status = (
            "completed"
            if rollout.terminated
            else rollout.policy_stop.status
            if rollout.policy_stop is not None
            else "limit_reached"
        )
        models = tuple(dict.fromkeys(item.decision.model for item in rollout.transitions))
        if not models and rollout.policy_stop is not None:
            models = (rollout.policy_stop.model,)
        return Generation(
            rollout.transitions[-1].result.observation if status == "completed" else "",
            status=status,
            stop_reason=rollout.stop_reason,
            steps=(Step("tool_result", encoded, name="environment_rollout"),),
            model=" | ".join(models) or "unknown",
        )


class EnvironmentReturnEvaluator:
    """Score trusted RolloutGenerator evidence, never an agent's reward claim.

    The reward is the sum of recorded environment rewards. Incomplete rollouts
    receive zero for Gym adaptation, while their original rewards stay in the
    trajectory. A terminal failure is still a completed task with its actual reward.
    Do not use this evaluator to authenticate arbitrary third-party generations.
    """

    def __init__(self, *, name: str = "environment-return-v1") -> None:
        _text(name, "evaluator name")
        self.name = name

    async def evaluate(self, task: Task, generation: Generation) -> Reward:
        evidence = [
            step
            for step in generation.steps
            if step.kind == "tool_result" and step.name == "environment_rollout"
        ]
        if len(evidence) != 1:
            raise ValueError("expected exactly one trusted environment rollout")
        rollout = Rollout.from_json(evidence[0].content)
        if rollout.task != task:
            raise ValueError("rollout task does not match the evaluated task")
        if generation.status == "completed":
            if (
                not rollout.terminated
                or generation.output != rollout.transitions[-1].result.observation
            ):
                raise ValueError("completed generation must match its terminal rollout")
            value = rollout.total_reward
        else:
            if rollout.terminated:
                raise ValueError("incomplete generation contradicts a terminal rollout")
            value = 0.0
        return Reward(
            value,
            self.name,
            f"Environment return {rollout.total_reward:g}; stop={rollout.stop_reason}; "
            f"transitions={len(rollout.transitions)}.",
        )
