"""Build private/local Material guides without deleting or replacing output files.

README becomes index.md; docs, assets and other root Markdown retain their paths.
Stages and failed output are intentionally retained for inspection, never removed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


def _linked(path: Path) -> bool:
    """Reject symlinks and Windows junction/reparse points in source/output paths."""
    info = path.lstat()
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _reserve(output: Path, repo: Path) -> Path:
    for parent in (output, *output.parents):
        if parent.exists() and _linked(parent):
            raise ValueError("output and its parents must not be symlinks or junctions")
    output = output.resolve()
    for protected in ("docs", "assets", ".git"):
        if output.is_relative_to(repo / protected):
            raise ValueError("output must not be inside documentation sources or .git")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".guides-build-reservation"
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write("Reserved guide output. Completion requires build-complete.json.\n")
    except FileExistsError:
        raise ValueError("output is already reserved by another build") from None
    if any(item != marker for item in output.iterdir()):
        raise ValueError("output changed while being reserved; existing files were preserved")
    return output


def _files(folder: Path):
    if _linked(folder):
        raise ValueError("documentation sources must not be symlinks or junctions")
    for item in sorted(folder.iterdir()):
        if _linked(item):
            raise ValueError("documentation sources must not be symlinks or junctions")
        if item.is_dir():
            yield from _files(item)
        elif item.is_file():
            yield item


def _copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#\s+(.+?)\s*#*\s*$", line)
        if match:
            return match[1]
    return path.stem.replace("-", " ").replace("_", " ").title()


def _site_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(ord(char) < 33 for char in value)
    ):
        raise ValueError(
            "site URL must be an HTTP(S) origin/base path without credentials or query"
        )
    return value.rstrip("/") + "/"


def build_guides(repo_root: Path, output: Path, *, site_url: str | None = None) -> dict:
    """Build strictly in a fresh .work tree, then copy each output file exclusively.

    A reservation and any partial files remain on failure; retry with a fresh path.
    No filesystem tree is deleted. This operation does not host or publish a site.
    """
    import yaml

    repo = Path(repo_root).resolve()
    site_url = _site_url(site_url)
    for required in (repo / "README.md", repo / "mkdocs.yml", repo / "docs"):
        if not required.exists() or _linked(required):
            raise ValueError("repository must contain regular README.md, mkdocs.yml and docs")
    config = yaml.safe_load((repo / "mkdocs.yml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("mkdocs.yml must be a configuration mapping")
    work = repo / ".work"
    if work.exists() and (_linked(work) or not work.is_dir()):
        raise ValueError(".work must be a regular directory")
    output = _reserve(Path(output), repo)
    work.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="guides-build-", dir=work))
    content, site = stage / "content", stage / "site"
    content.mkdir()
    _copy_new(repo / "README.md", content / "index.md")
    for markdown in sorted(repo.glob("*.md")):
        if markdown.name != "README.md":
            if _linked(markdown):
                raise ValueError("root Markdown files must not be symlinks")
            _copy_new(markdown, content / markdown.name)
    for name in ("docs", "assets"):
        source = repo / name
        if source.exists():
            for path in _files(source):
                _copy_new(path, content / path.relative_to(repo))
    # Preserve links to the source README when it becomes the site's home page.
    for markdown in content.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")

        def home_link(match, current_dir=markdown.parent):
            relative = match[1] + "README.md"
            if (current_dir / relative).resolve() != content / "README.md":
                return match[0]
            return "](" + match[1] + "index.md"

        text = re.sub(r"\]\(((?:\.\./)*)README\.md(?=[)#])", home_link, text)
        markdown.write_text(text, encoding="utf-8")
    pages = sorted(content.rglob("*.md"))
    groups = {"Guides": [], "Project reference": [], "Artwork": []}
    for path in pages:
        relative = path.relative_to(content).as_posix()
        if relative == "index.md":
            continue
        group = (
            "Guides"
            if relative.startswith("docs/")
            else ("Artwork" if relative.startswith("assets/") else "Project reference")
        )
        groups[group].append({_title(path): relative})
    config.update(
        docs_dir=str(content),
        site_dir=str(site),
        strict=True,
        nav=[{"Home": "index.md"}] + [{name: links} for name, links in groups.items() if links],
    )
    if site_url is not None:
        config["site_url"] = site_url
    generated_config = stage / "mkdocs.yml"
    with generated_config.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    result = subprocess.run(
        # MkDocs cleans only this not-yet-created directory in our unique staging
        # tree. The caller's final output is never passed to MkDocs or cleaned.
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-f", str(generated_config)],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError("strict guide build failed:\n" + result.stdout + result.stderr)
    if not (site / "index.html").is_file():
        raise RuntimeError("guide build did not produce index.html")
    copied = []
    for source in _files(site):
        relative = source.relative_to(site)
        _copy_new(source, output / relative)
        copied.append(relative.as_posix())
    report = {
        "output": str(output),
        "staging": str(stage),
        "pages": len(pages),
        "files": len(copied),
        "site_url": site_url,
        "deployed": False,
    }
    with (output / "build-complete.json").open("x", encoding="utf-8") as handle:
        json.dump({"pages": len(pages), "files": len(copied), "deployed": False}, handle)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-url", help="optional canonical private-host URL; does not deploy")
    args = parser.parse_args(argv)
    try:
        report = build_guides(
            Path(__file__).resolve().parents[1], args.output, site_url=args.site_url
        )
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
