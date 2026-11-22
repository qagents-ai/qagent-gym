"""Dependency-free data boundaries for Inspect AI and GEPA.

These helpers export documented data shapes. They neither run an evaluator
service nor invoke an optimizer, and they never infer answer keys from model output.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from qagent_gym.records import Episode, Task


def write_inspect_dataset(
    tasks: Iterable[Task],
    targets: Mapping[str, str],
    path: str | Path,
    *,
    partition: Literal["validation", "test"] = "test",
    excluded_task_ids: Iterable[str] = (),
) -> int:
    """Write JSONL for Inspect's json_dataset, refusing to replace an existing file.

    Supply ground truth separately from tasks. Targets must match the task IDs
    exactly. Exclude the IDs used for training and, for a final test, optimizer
    validation. Different IDs can still describe overlapping tasks: callers must
    curate genuinely separate datasets. All inputs are checked before opening path.
    """
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("an evaluation dataset must not be empty")
    if any(not isinstance(task, Task) or task.split != "eval" for task in tasks):
        raise ValueError("Inspect evaluation export requires eval tasks")
    if partition not in ("validation", "test"):
        raise ValueError("partition must be validation or test")
    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs must be unique")
    if isinstance(excluded_task_ids, str | bytes):
        raise ValueError("excluded task IDs must be a collection, not one string")
    excluded_ids = tuple(excluded_task_ids)
    if any(not isinstance(item, str) or not item.strip() for item in excluded_ids):
        raise ValueError("excluded task IDs must be nonempty strings")
    excluded = set(excluded_ids)
    if excluded.intersection(ids):
        raise ValueError("evaluation task IDs overlap excluded training/validation IDs")
    if not isinstance(targets, Mapping) or set(targets) != set(ids):
        raise ValueError("targets must match task IDs exactly")
    if any(not isinstance(target, str) for target in targets.values()):
        raise ValueError("each target must be a string")
    rows = [
        {
            "id": task.task_id,
            "input": task.prompt,
            "target": targets[task.task_id],
            "metadata": {"qagent_gym_split": task.split, "partition": partition},
        }
        for task in tasks
    ]
    encoded = b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        for row in rows
    )
    with Path(path).open("xb") as stream:
        stream.write(encoded)
    return len(rows)


def gepa_reflective_dataset(
    episodes: Iterable[Episode], *, component: str = "playbook"
) -> dict[str, list[dict[str, Any]]]:
    """Build one component's reflective records for a GEPA adapter.

    Only completed training episodes are accepted. Preserve episode order and
    require distinct run IDs and one evaluator so scores keep a common meaning.
    Higher scores must mean better performance for the chosen evaluator.
    The result matches make_reflective_dataset's data shape; GEPA does not supply
    a native loader for this JSON. Candidate evaluation and optimization require
    a separate adapter. Held-out feedback must never enter this export.
    """
    if not isinstance(component, str) or not component.strip():
        raise ValueError("component must be a nonempty string")
    episodes = tuple(episodes)
    if not episodes:
        raise ValueError("reflection requires at least one training episode")
    if any(not isinstance(episode, Episode) for episode in episodes):
        raise ValueError("reflection requires Episode values")
    if any(episode.task.split != "train" for episode in episodes):
        raise ValueError("held-out episodes must not enter optimizer reflection")
    if any(episode.generation.status != "completed" for episode in episodes):
        raise ValueError("reflection requires completed generations")
    if len({episode.run_id for episode in episodes}) != len(episodes):
        raise ValueError("episode IDs must be unique")
    if len({episode.reward.evaluator for episode in episodes}) != 1:
        raise ValueError("reflection scores must come from one evaluator")
    return {
        component: [
            {
                "Inputs": {
                    "task": episode.task.prompt,
                    "learned_context": episode.context.render(),
                },
                "Generated Outputs": episode.generation.output,
                "Feedback": episode.reward.feedback,
                "score": episode.reward.value,
                "episode_id": episode.run_id,
                "evaluator": episode.reward.evaluator,
                "model": episode.generation.model,
                "context_revision": episode.context.revision,
            }
            for episode in episodes
        ]
    }
