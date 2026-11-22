"""Versioned episode exchange. Exporting data does not train model weights."""

from collections.abc import Iterable
from pathlib import Path

from .records import Episode


def export_jsonl(episodes: Iterable[Episode], path: str | Path) -> int:
    """Write a new export; refuse accidental replacement of an existing file."""
    records = tuple(episodes)
    encoded = [episode.to_json() for episode in records]
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for line in encoded:
            stream.write(line + "\n")
    return len(records)


def read_jsonl(path: str | Path) -> tuple[Episode, ...]:
    episodes = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                episodes.append(Episode.from_json(line))
            except ValueError as exc:
                raise ValueError(f"line {number}: {exc}") from exc
    return tuple(episodes)
