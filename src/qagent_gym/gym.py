"""The learning loop; evaluators and peer reviewers remain explicit dependencies."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Protocol

from .records import Context, Episode, Generation, Lesson, MemoryRecord, PlaybookDelta, Reward, Task
from .storage import SQLiteStore


class Generator(Protocol):
    async def generate(self, task: Task, context: Context) -> Generation: ...


class Evaluator(Protocol):
    """May use an objective checker, externally supplied feedback, or a peer agent."""

    async def evaluate(self, task: Task, generation: Generation) -> Reward: ...


class Reflector(Protocol):
    async def reflect(self, episode: Episode) -> tuple[Lesson, ...]: ...


class Curator(Protocol):
    def curate(
        self, episode: Episode, lessons: tuple[Lesson, ...]
    ) -> tuple[PlaybookDelta, ...]: ...


class MemoryStore(Protocol):
    def add(self, record: MemoryRecord) -> None: ...

    def search(self, query: str, *, namespace: str, limit: int = 5) -> tuple[MemoryRecord, ...]: ...


class DeltaCurator:
    """Deterministically turn validated lessons into replayable changes.

    Adding an existing bullet becomes an update. Contradictory duplicate bullet
    IDs in one reflection are rejected instead of silently picking a winner.
    """

    def curate(self, episode: Episode, lessons: tuple[Lesson, ...]) -> tuple[PlaybookDelta, ...]:
        seen = set()
        existing = {bullet.bullet_id for bullet in episode.context.bullets}
        changes = []
        for index, lesson in enumerate(lessons):
            if not isinstance(lesson, Lesson):
                raise ValueError("reflector must return Lesson values")
            if lesson.bullet_id in seen:
                raise ValueError("duplicate bullet_id in reflection")
            seen.add(lesson.bullet_id)
            operation = lesson.operation
            if operation == "add" and lesson.bullet_id in existing:
                operation = "update"
            changes.append(
                PlaybookDelta(
                    delta_id=f"{episode.run_id}:{index}",
                    bullet_id=lesson.bullet_id,
                    text=lesson.text,
                    source_episode_id=episode.run_id,
                    operation=operation,
                    helpful=lesson.helpful,
                    harmful=lesson.harmful,
                )
            )
        return tuple(changes)


class ExactMatchEvaluator:
    """Keep answer keys outside generator input; include feedback only after generation."""

    def __init__(self, answers: Mapping[str, str], *, name: str = "exact-match-v1") -> None:
        self._answers = dict(answers)
        self.name = name

    async def evaluate(self, task: Task, generation: Generation) -> Reward:
        if generation.status != "completed":
            return Reward(0.0, self.name, f"Run stopped: {generation.status}.")
        expected = self._answers[task.task_id]
        matched = generation.output.strip() == expected.strip()
        return Reward(
            float(matched),
            self.name,
            "Correct." if matched else f"Expected {expected!r}; got {generation.output!r}.",
        )


class Gym:
    """Train context between episodes, or evaluate against a frozen snapshot.

    Serialize training jobs per namespace. Distinct concurrent updates to the same
    bullet use last-writer-wins; idempotency protects repeated delta IDs. Model/service
    exceptions propagate. Episodes are durable before adaptation, but the whole
    generation/learning process is not a crash-resumable transaction.
    """

    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        reflector: Reflector,
        store: SQLiteStore,
        *,
        curator: Curator | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator
        self.reflector = reflector
        self.store = store
        self.curator = curator or DeltaCurator()
        self.memory = memory if memory is not None else store

    def _context(self, task: Task, snapshot: Context) -> Context:
        memories = self.memory.search(task.prompt, namespace=snapshot.namespace, limit=5)
        return replace(snapshot, memories=memories)

    async def _episode(self, task: Task, context: Context) -> Episode:
        generation = await self.generator.generate(task, context)
        reward = await self.evaluator.evaluate(task, generation)
        episode = Episode(uuid.uuid4().hex, task, generation, reward, context)
        self.store.save_episode(episode)
        return episode

    async def train(self, task: Task, *, namespace: str) -> Episode:
        if task.split != "train":
            raise ValueError("training requires split='train'")
        context = self._context(task, self.store.snapshot(namespace))
        episode = await self._episode(task, context)
        if episode.generation.status == "completed":
            lessons = tuple(await self.reflector.reflect(episode))
            self.store.apply(namespace, self.curator.curate(episode, lessons))
            self.memory.add(
                MemoryRecord(
                    memory_id=episode.run_id,
                    namespace=namespace,
                    text=f"Task: {task.prompt}\nAnswer: {episode.generation.output}\n"
                    f"Feedback: {episode.reward.feedback}",
                    source_episode_id=episode.run_id,
                )
            )
        return episode

    async def evaluate(self, tasks: Iterable[Task], *, namespace: str) -> tuple[Episode, ...]:
        """Snapshot every input context before generation. Write episode logs only."""
        tasks = tuple(tasks)
        if any(task.split != "eval" for task in tasks):
            raise ValueError("evaluation requires split='eval'")
        snapshot = self.store.snapshot(namespace)
        contexts = tuple(self._context(task, snapshot) for task in tasks)
        return tuple(
            [
                await self._episode(task, context)
                for task, context in zip(tasks, contexts, strict=True)
            ]
        )
