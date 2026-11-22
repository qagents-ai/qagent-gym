"""Small offline CLI. State and exports are written only to explicit local paths."""

import argparse
import asyncio
import json
from pathlib import Path

from .demo import run_demo
from .export import export_jsonl
from .storage import SQLiteStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qagent-gym")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a scripted qagents learning demonstration")
    demo.add_argument("--db", type=Path, default=Path(".gym/demo.sqlite3"))
    demo.add_argument("--namespace", default="metres-demo")
    demo.add_argument("--export", type=Path)
    export = commands.add_parser("export", help="Export recorded episodes as versioned JSONL")
    export.add_argument("--db", type=Path, required=True)
    export.add_argument("--namespace", required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            if args.export and args.export.exists():
                parser.error("export path already exists; choose a new file")
            report = asyncio.run(
                run_demo(args.db, export_path=args.export, namespace=args.namespace)
            )
        else:
            if not args.db.is_file():
                parser.error("database does not exist")
            with SQLiteStore(args.db) as store:
                count = export_jsonl(store.episodes(args.namespace), args.output)
            report = {"exported_episodes": count, "output": str(args.output)}
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0
