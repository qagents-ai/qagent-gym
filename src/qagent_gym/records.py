"""Immutable, JSON-compatible public values. No provider objects enter the journal."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Literal


def _text(value: str, field: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{field} must be {'a string' if empty else 'a nonempty string'}")


def _count(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def _tuple(instance: object, field: str, cls: type) -> None:
    values = tuple(getattr(instance, field))
    if not all(isinstance(item, cls) for item in values):
        raise ValueError(f"{field} must contain {cls.__name__} values")
    object.__setattr__(instance, field, values)


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    split: Literal["train", "eval"] = "train"

    def __post_init__(self) -> None:
        _text(self.task_id, "task_id")
        _text(self.prompt, "prompt")
        if self.split not in ("train", "eval"):
            raise ValueError("split must be train or eval")


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    namespace: str
    text: str
    source_episode_id: str

    def __post_init__(self) -> None:
        for field in ("memory_id", "namespace", "text", "source_episode_id"):
            _text(getattr(self, field), field)


@dataclass(frozen=True)
class Bullet:
    bullet_id: str
    text: str
    source_episode_id: str
    helpful: int = 0
    harmful: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        for field in ("bullet_id", "text", "source_episode_id"):
            _text(getattr(self, field), field)
        _count(self.helpful, "helpful")
        _count(self.harmful, "harmful")
        if type(self.active) is not bool:
            raise ValueError("active must be boolean")


@dataclass(frozen=True)
class Context:
    namespace: str
    revision: int = 0
    bullets: tuple[Bullet, ...] = ()
    memories: tuple[MemoryRecord, ...] = ()

    def __post_init__(self) -> None:
        _text(self.namespace, "namespace")
        _count(self.revision, "revision")
        _tuple(self, "bullets", Bullet)
        _tuple(self, "memories", MemoryRecord)
        if any(item.namespace != self.namespace for item in self.memories):
            raise ValueError("context memories must share its namespace")

    def render(self) -> str:
        """Render data for explicit injection into an agent's task prompt."""
        lessons = "\n".join(f"[{b.bullet_id}] {b.text}" for b in self.bullets if b.active)
        memories = "\n".join(f"[{m.memory_id}] {m.text}" for m in self.memories)
        return (
            f"Learned playbook:\n{lessons or '(empty)'}\n\n"
            f"Retrieved memory:\n{memories or '(empty)'}"
        )


@dataclass(frozen=True)
class Step:
    kind: Literal["message", "tool_call", "tool_result"]
    content: str
    role: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("message", "tool_call", "tool_result"):
            raise ValueError("invalid step kind")
        for field in ("content", "role", "name"):
            _text(getattr(self, field), field, empty=True)


@dataclass(frozen=True)
class Generation:
    output: str
    status: str = "completed"
    steps: tuple[Step, ...] = ()
    model: str = "unknown"
    stop_reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("completed", "paused", "limit_reached", "cancelled", "blocked"):
            raise ValueError("invalid generation status")
        _text(self.output, "output", empty=True)
        _text(self.model, "model")
        _text(self.stop_reason, "stop_reason", empty=True)
        _tuple(self, "steps", Step)
        if self.status != "completed" and self.output:
            raise ValueError("incomplete generations must not expose an answer")


@dataclass(frozen=True)
class Reward:
    value: float
    evaluator: str
    feedback: str

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("reward value must be numeric")
        try:
            finite = math.isfinite(self.value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("reward value must be finite")
        _text(self.evaluator, "evaluator")
        _text(self.feedback, "feedback", empty=True)


@dataclass(frozen=True)
class Lesson:
    bullet_id: str
    text: str
    operation: Literal["add", "update", "deprecate"] = "add"
    helpful: int = 0
    harmful: int = 0

    def __post_init__(self) -> None:
        _validate_change(self)


@dataclass(frozen=True)
class PlaybookDelta:
    delta_id: str
    bullet_id: str
    text: str
    source_episode_id: str
    operation: Literal["add", "update", "deprecate"] = "add"
    helpful: int = 0
    harmful: int = 0

    def __post_init__(self) -> None:
        _text(self.delta_id, "delta_id")
        _text(self.source_episode_id, "source_episode_id")
        _validate_change(self)


def _validate_change(value: Lesson | PlaybookDelta) -> None:
    if value.operation not in ("add", "update", "deprecate"):
        raise ValueError("invalid delta operation")
    _text(value.bullet_id, "bullet_id")
    _text(value.text, "text", empty=value.operation == "deprecate")
    _count(value.helpful, "helpful")
    _count(value.harmful, "harmful")


@dataclass(frozen=True)
class Episode:
    run_id: str
    task: Task
    generation: Generation
    reward: Reward
    context: Context
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported episode schema_version")
        for field, cls in (
            ("task", Task),
            ("generation", Generation),
            ("reward", Reward),
            ("context", Context),
        ):
            if not isinstance(getattr(self, field), cls):
                raise ValueError(f"{field} must be {cls.__name__}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> Episode:
        """Decode version 1 records, rejecting malformed values and unknown fields."""
        try:
            data = json.loads(text)
            generation = data["generation"]
            generation["steps"] = tuple(Step(**value) for value in generation["steps"])
            context = data["context"]
            context["bullets"] = tuple(Bullet(**value) for value in context["bullets"])
            context["memories"] = tuple(MemoryRecord(**value) for value in context["memories"])
            return cls(
                **{
                    **data,
                    "task": Task(**data["task"]),
                    "generation": Generation(**generation),
                    "reward": Reward(**data["reward"]),
                    "context": Context(**context),
                }
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid episode JSON: {exc}") from exc
