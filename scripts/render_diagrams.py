"""Render D2 and Mermaid diagram sources to static SVGs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def render(browser: Path, *, previews: Path | None = None) -> list[str]:
    repo = Path(__file__).resolve().parents[1]
    cli = repo / ".work/diagram-tools/node_modules/@mermaid-js/mermaid-cli/src/cli.js"
    node = shutil.which("node")
    if node is None or not cli.is_file() or not browser.is_file():
        raise ValueError("install Mermaid CLI under .work/diagram-tools and provide a browser path")
    work = repo / ".work"
    config_dir = Path(tempfile.mkdtemp(prefix="diagram-render-", dir=work))
    puppet = config_dir / "puppeteer.json"
    puppet.write_text(json.dumps({"executablePath": str(browser.resolve()), "headless": True}))
    source_dir = repo / "docs/diagrams"
    output_dir = repo / "assets/diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    if previews is not None:
        previews.mkdir(parents=True, exist_ok=True)
    rendered = []
    for source in sorted(source_dir.glob("*.mmd")):
        targets = [output_dir / (source.stem + ".svg")]
        if previews is not None:
            targets.append(previews / (source.stem + ".png"))
        for target in targets:
            subprocess.run(
                [
                    node,
                    str(cli),
                    "--input",
                    str(source),
                    "--output",
                    str(target),
                    "--configFile",
                    str(source_dir / "theme.json"),
                    "--puppeteerConfigFile",
                    str(puppet),
                    "--svgId",
                    source.stem,
                    "--backgroundColor",
                    "white",
                    "--size",
                    "900" if target.suffix == ".png" else "1000",
                    "--quiet",
                ],
                check=True,
                timeout=90,
            )
        rendered.append(source.stem)
    return rendered


def render_d2(executable: Path) -> list[str]:
    repo = Path(__file__).resolve().parents[1]
    if not executable.is_file():
        raise ValueError("provide the D2 executable path")
    output_dir = repo / "assets/diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for source in sorted((repo / "docs/diagrams").glob("*.d2")):
        subprocess.run(
            [
                str(executable.resolve()),
                "--layout=elk",
                "--pad=24",
                source.name,
                str(output_dir / (source.stem + ".svg")),
            ],
            cwd=source.parent,
            check=True,
            timeout=90,
        )
        rendered.append(source.stem)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, help="Chrome or Chromium path for Mermaid")
    parser.add_argument("--d2", type=Path, help="D2 executable path")
    parser.add_argument("--previews", type=Path)
    args = parser.parse_args()
    if args.browser is None and args.d2 is None:
        parser.error("provide --browser for Mermaid, --d2 for D2, or both")
    if args.previews and args.browser is None:
        parser.error("--previews requires --browser")
    mermaid = render(args.browser, previews=args.previews) if args.browser else []
    d2 = render_d2(args.d2) if args.d2 else []
    print(json.dumps({"rendered": mermaid + d2}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
