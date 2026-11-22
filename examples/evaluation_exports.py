"""Export real Gym episodes to separate optimizer-reflection and evaluation data.

Run: python -m examples.evaluation_exports --db .gym/export-recipe.sqlite3
No Inspect evaluator or GEPA optimizer is imported or run by this recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

from qagent_gym.integrations.evaluation import gepa_reflective_dataset, write_inspect_dataset
from qagent_gym.storage import SQLiteStore

from . import before_after
from ._support import run_cli


async def run(path: str | Path, *, namespace: str = "evaluation-exports") -> dict:
    path = Path(path)
    inspect_path = path.with_suffix(".inspect.jsonl")
    gepa_path = path.with_suffix(".gepa.json")
    if inspect_path.exists() or gepa_path.exists():
        raise ValueError("export files already exist; choose a fresh database path")
    comparison = await before_after.run(path, namespace=namespace)
    with SQLiteStore(path) as store:
        episodes = store.episodes(namespace)
    training = [episode for episode in episodes if episode.task.split == "train"]
    # Before/after evaluation logged each held-out task twice. Export each task
    # once and use the evaluator's ground truth, never the agent's answer.
    heldout = {
        episode.task.task_id: episode.task for episode in episodes if episode.task.split == "eval"
    }
    targets = {row["task_id"]: row["expected"] for row in comparison["rows"]}
    count = write_inspect_dataset(
        heldout.values(),
        targets,
        inspect_path,
        partition="test",
        excluded_task_ids=[episode.task.task_id for episode in training],
    )
    reflection = gepa_reflective_dataset(training, component="playbook")
    with gepa_path.open("x", encoding="utf-8") as stream:
        json.dump(reflection, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "inspect_samples": count,
        "gepa_training_examples": len(reflection["playbook"]),
        "inspect_path": str(inspect_path),
        "gepa_path": str(gepa_path),
        "score_before": comparison["score_before"],
        "score_after": comparison["score_after"],
        "execution": "dataset export only; no Inspect run or GEPA optimization",
    }


if __name__ == "__main__":
    run_cli(run, __doc__)
