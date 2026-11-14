"""Create Matplotlib distributions directly from retained raw trial observations."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from pathlib import Path

from .protocol import read_json


def plot(results: Path) -> tuple[Path, Path]:
    records = read_json(results / "raw.json")
    metadata = read_json(results / "metadata.json")
    if not records or not isinstance(records, list):
        raise ValueError("raw.json must contain observations")
    outputs = results / "recovery-distributions.svg", results / "recovery-distributions.png"
    if any(path.exists() for path in outputs):
        raise FileExistsError("plot outputs already exist")
    cache = results / "matplotlib-cache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"qagents": "#137C86", "qagents-effects": "#137C86", "langgraph": "#6652A3"}
    frameworks = [name for name in colors if any(row["framework"] == name for row in records)]
    offsets = (
        [0]
        if len(frameworks) == 1
        else [0.34 * (index / (len(frameworks) - 1) - 0.5) for index in range(len(frameworks))]
    )
    faults = ("before_effect", "after_commit_before_ack", "after_checkpoint")
    labels = ("Before effect", "Commit, no acknowledgement", "After checkpoint")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for axis, metric, title in zip(
        axes,
        ("recovery_launch_to_result_ms", "lease_acquisition_wait_ms"),
        ("Recovery process launch → completed result", "Runtime lease acquisition wait"),
        strict=True,
    ):
        for index, fault in enumerate(faults):
            for offset, framework in zip(offsets, frameworks, strict=True):
                values = [
                    record["timings"][metric]
                    for record in records
                    if record["framework"] == framework
                    and record["fault"] == fault
                    and record["deduplicate"]
                    and record["outcome"] == "passed"
                ]
                if not values:
                    continue
                if any(
                    not isinstance(value, int | float) or not math.isfinite(value)
                    for value in values
                ):
                    raise ValueError("nonfinite timing in raw observations")
                position = index + offset
                axis.boxplot(
                    [values],
                    positions=[position],
                    widths=0.24,
                    showfliers=False,
                    patch_artist=True,
                    boxprops={"facecolor": colors[framework], "alpha": 0.18},
                    medianprops={"color": colors[framework], "linewidth": 2},
                    whiskerprops={"color": colors[framework]},
                    capprops={"color": colors[framework]},
                )
                jitter = [(number - (len(values) - 1) / 2) * 0.018 for number in range(len(values))]
                axis.scatter(
                    [position + delta for delta in jitter],
                    values,
                    color=colors[framework],
                    s=28,
                    alpha=0.8,
                    zorder=3,
                )
                axis.annotate(
                    f"n={len(values)}",
                    (position, max(values)),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=colors[framework],
                )
        axis.set_xticks(range(3), labels, fontsize=9)
        axis.set_title(title, loc="left", fontsize=12, pad=12)
        axis.set_ylabel("Milliseconds · lower is shorter, not necessarily safer")
        axis.set_ylim(bottom=-5)
        axis.margins(y=0.2)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    outcomes = Counter(record["outcome"] for record in records)
    figure.suptitle(
        "Local recovery measurements · deterministic credit workload\n"
        + metadata["configuration"]["label"],
        fontsize=15,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    figure.legend(
        handles=[Patch(facecolor=colors[name], label=name) for name in frameworks],
        loc="outside lower center",
        ncols=2,
        frameon=False,
        title=(
            "SQLite WAL / FULL; LangGraph sync. Dots are individual positive trials. "
            "Lease policy differs.\n"
            + "; ".join(f"{key}: {count}" for key, count in sorted(outcomes.items()))
            + ". Controls and incomplete observations are excluded from timing plots."
        ),
    )
    for path in outputs:
        with path.open("xb") as stream:
            figure.savefig(stream, format=path.suffix[1:], dpi=180, facecolor="white")
    plt.close(figure)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    print("\n".join(str(path) for path in plot(parser.parse_args().results)))
