import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mkdocs")
pytest.importorskip("material")
yaml = pytest.importorskip("yaml")

from scripts import build_guides as builder  # noqa: E402


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docs" / "nested").mkdir(parents=True)
    (repo / "assets").mkdir()
    (repo / "README.md").write_text(
        "# Sample Gym\n\n[Guide](docs/first.md) [Spec](SPEC.md)\n\n![Art](assets/art.svg)\n"
    )
    (repo / "SPEC.md").write_text("# Contracts\n\n[Home](README.md)\n")
    (repo / "docs" / "first.md").write_text("# First guide\n\n[Next](nested/new.md)\n")
    (repo / "docs" / "nested" / "new.md").write_text("# A newly added guide\n")
    (repo / "assets" / "art.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (repo / "assets" / "README.md").write_text("# Artwork\n\n[Home](../README.md)\n")
    (repo / "mkdocs.yml").write_bytes(
        (Path(__file__).resolve().parents[1] / "mkdocs.yml").read_bytes()
    )
    return repo


def fake_build(command, **kwargs):
    config = yaml.safe_load(Path(command[-1]).read_text(encoding="utf-8"))
    site = Path(config["site_dir"])
    site.mkdir()
    (site / "index.html").write_text("<html>Complete fixture build</html>")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_native_strict_build_stages_links_assets_and_every_guide(repository, tmp_path):
    original_readme = (repository / "README.md").read_bytes()
    output = tmp_path / "site"
    report = builder.build_guides(repository, output, site_url="https://private.example/gym")
    assert report["pages"] == 5
    assert report["site_url"] == "https://private.example/gym/"
    assert report["deployed"] is False
    assert (output / "index.html").is_file()
    assert (output / "docs" / "nested" / "new" / "index.html").is_file()
    assert (output / "assets" / "art.svg").is_file()
    assert (output / "search" / "search_index.json").is_file()
    assert (repository / "README.md").read_bytes() == original_readme
    stage = Path(report["staging"])
    assert stage.parent == repository / ".work"
    staged_spec = (stage / "content" / "SPEC.md").read_text()
    assert "[Home](index.md)" in staged_spec
    assert "[Home](../index.md)" in (stage / "content" / "assets" / "README.md").read_text()
    config = yaml.safe_load((stage / "mkdocs.yml").read_text())
    assert "docs/nested/new.md" in json.dumps(config["nav"])
    assert "assets/README.md" in json.dumps(config["nav"])
    assert json.loads((output / "build-complete.json").read_text())["pages"] == 5


def test_existing_output_is_preserved_without_invoking_builder(repository, tmp_path, monkeypatch):
    output = tmp_path / "site"
    output.mkdir()
    sentinel = output / "index.html"
    sentinel.write_bytes(b"previous private site")
    monkeypatch.setattr(builder.subprocess, "run", lambda *a, **k: pytest.fail("build invoked"))
    with pytest.raises(ValueError, match="new or empty"):
        builder.build_guides(repository, output)
    assert sentinel.read_bytes() == b"previous private site"
    assert list(output.iterdir()) == [sentinel]


def test_nested_readme_link_keeps_its_original_target(repository, tmp_path, monkeypatch):
    (repository / "docs" / "README.md").write_text("# Documentation overview\n")
    (repository / "docs" / "first.md").write_text("# Guide\n\n[Overview](README.md)\n")
    monkeypatch.setattr(builder.subprocess, "run", fake_build)
    result = builder.build_guides(repository, tmp_path / "site")
    staged = Path(result["staging"]) / "content" / "docs" / "first.md"
    assert "[Overview](README.md)" in staged.read_text()


@pytest.mark.parametrize("already_empty", [False, True])
def test_competing_build_cannot_enter_reserved_destination(
    repository, tmp_path, monkeypatch, already_empty
):
    output = tmp_path / "site"
    if already_empty:
        output.mkdir()
    entered = []

    def compete(command, **kwargs):
        entered.append(True)
        with pytest.raises(ValueError, match="new or empty"):
            builder.build_guides(repository, output)
        return fake_build(command, **kwargs)

    monkeypatch.setattr(builder.subprocess, "run", compete)
    builder.build_guides(repository, output)
    assert entered == [True]
    assert (output / "index.html").read_text() == "<html>Complete fixture build</html>"


def test_file_injected_during_build_is_preserved(repository, tmp_path, monkeypatch):
    output = tmp_path / "site"

    def inject(command, **kwargs):
        (output / "index.html").write_bytes(b"owned by another writer")
        return fake_build(command, **kwargs)

    monkeypatch.setattr(builder.subprocess, "run", inject)
    with pytest.raises(FileExistsError):
        builder.build_guides(repository, output)
    assert (output / "index.html").read_bytes() == b"owned by another writer"
    assert not (output / "build-complete.json").exists()


def test_strict_missing_link_failure_retains_reservation_without_publishing(repository, tmp_path):
    (repository / "docs" / "first.md").write_text("# Guide\n\n[Missing](absent.md)\n")
    output = tmp_path / "site"
    with pytest.raises(RuntimeError, match="strict guide build failed"):
        builder.build_guides(repository, output)
    assert {path.name for path in output.iterdir()} == {".guides-build-reservation"}


@pytest.mark.parametrize("relative", ["docs/generated", "assets/generated", ".git/generated"])
def test_output_cannot_be_inside_source_or_git(repository, relative):
    with pytest.raises(ValueError, match="must not be inside"):
        builder.build_guides(repository, repository / relative)
    assert not (repository / relative).exists()


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "https://name:secret@host/", "https://host/?q=x"]
)
def test_invalid_site_url_rejected_before_output_creation(repository, tmp_path, url):
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="site URL"):
        builder.build_guides(repository, output, site_url=url)
    assert not output.exists()
