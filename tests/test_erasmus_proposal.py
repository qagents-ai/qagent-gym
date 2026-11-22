import asyncio
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples import erasmus_proposal as erasmus
from qagent_gym.storage import SQLiteStore


@pytest.fixture
def brief():
    return json.loads(erasmus.DEFAULT_BRIEF.read_text(encoding="utf-8"))


@pytest.fixture
def completed(tmp_path):
    output = tmp_path / "proposal"
    report = asyncio.run(erasmus.run(output))
    return output, report


def test_four_real_agents_review_correct_and_persist_a_complete_proposal(completed, brief):
    output, report = completed
    assert report["agent_calls"] == {"planner": 1, "writer": 2, "reviewer": 2, "reflector": 1}
    assert report["initial_error_count"] == 7
    assert report["final_error_count"] == 0
    assert report["remaining_evidence_warnings"] == 6
    assert report["learning_revision"] == 1
    assert report["frozen_final_review"] is True
    assert report["internal_checks_passed"] is True
    assert report["submission_ready"] is False
    assert {file.name for file in output.iterdir()} == {
        ".run-reservation",
        "draft-initial.md",
        "proposal.md",
        "brief.json",
        "plan.json",
        "review-before.json",
        "review-after.json",
        "workflow.json",
        "episodes.jsonl",
        "gym.sqlite3",
    }
    proposal = (output / "proposal.md").read_text(encoding="utf-8")
    initial = (output / "draft-initial.md").read_text(encoding="utf-8")
    assert "Requested grant: EUR 260,000" in initial
    assert "## Risks and safeguards" not in initial
    assert "Requested grant: EUR 250,000" in proposal
    assert "Estimated total project cost: EUR 275,000" in proposal
    assert "| Total | 84,000 | 82,000 | 84,000 | 250,000 |" in proposal
    assert "Planned unique learners: 180" in proposal
    assert "Planned completions: 144" in proposal
    assert "Baseline competence and current provider practice are not known" in proposal
    assert len(proposal.split()) > 2500
    for heading in erasmus.SECTIONS:
        assert proposal.count(f"## {heading}\n") == 1
    for wp in brief["work_packages"]:
        for activity in wp["activities"]:
            assert activity["deliverable"] in proposal
            assert activity["acceptance"] in proposal
    review = erasmus.review_proposal(proposal, brief)
    assert review == json.loads((output / "review-after.json").read_text())
    assert review["checks_run"] >= 70
    with SQLiteStore(output / "gym.sqlite3") as store:
        first, final = store.episodes("erasmus-proposal")
        assert first.reward.value == 0
        assert final.reward.value == 1
        assert first.task.split == "train" and final.task.split == "eval"
        assert first.context.bullets == ()
        assert final.context.bullets[0].text == erasmus.CORRECTION_RULE
        assert final.context.bullets[0].source_episode_id == first.run_id
        assert len(store.changes("erasmus-proposal")) == 1
        assert {m.source_episode_id for m in store.memories("erasmus-proposal")} == {first.run_id}
        # The held context is frozen; this is revision QA on the same brief, not a heldout benchmark.
        assert first.task.prompt == final.task.prompt
        assert first.generation.status == final.generation.status == "completed"
        assert any(
            step.kind == "message" and step.role == "assistant" and step.content == proposal
            for step in final.generation.steps
        )


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("Requested grant: EUR 250,000", "Requested grant: EUR 400,000", "claim_mismatch"),
        ("| 45,000 | 50,000 | 5,000 |", "| 44,000 | 50,000 | 5,000 |", "budget_mismatch"),
        ("| WP4 | 11 | 20 |", "| WP4 | 10 | 20 |", "timeline_mismatch"),
        (
            "| WP1 | 25,000 | 10,000 | 10,000 |",
            "| WP1 | 24,000 | 11,000 | 10,000 |",
            "allocation_mismatch",
        ),
        ("| Adult completions | 144 |", "| Adult completions | 180 |", "indicator_mismatch"),
        ("| R1 |", "| OMITTED |", "missing_risk"),
        (erasmus.SOURCES["countries"], "https://example.invalid/source", "missing_citation"),
        ("## Objectives and target groups", "## Omitted objectives", "missing_section"),
    ],
)
def test_reviewer_checks_actual_draft_content(brief, old, new, error):
    original = erasmus.render_proposal(brief, erasmus.compute_plan(brief))
    assert old in original
    changed = original.replace(old, new)
    report = erasmus.review_proposal(changed, brief)
    assert not report["internal_checks_passed"]
    assert error in {item["code"] for item in report["errors"]}


def test_evidence_gaps_cannot_disappear_behind_a_clean_budget(brief):
    proposal = erasmus.render_proposal(brief, erasmus.compute_plan(brief))
    proposal = proposal.replace(brief["missing_evidence"][0]["reason"], "Already approved")
    report = erasmus.review_proposal(proposal, brief)
    assert "missing_evidence_notice" in {item["code"] for item in report["errors"]}
    assert report["submission_ready"] is False


def test_changed_brief_recalculates_objectives_activities_and_indicators(tmp_path, brief):
    brief["title"] = "Accessible Digital Practice in Community Centres"
    brief["learners_per_partner"] = 40
    brief["facilitators_per_partner"] = 8
    brief["completion_percent"] = 75
    report = asyncio.run(erasmus.run(tmp_path / "adapted", brief=brief))
    document = Path(report["proposal_path"]).read_text(encoding="utf-8")
    assert report["learners"] == 120
    assert report["facilitators"] == 24
    assert report["final_error_count"] == 0
    assert brief["title"] in document
    assert "Planned completions: 90" in document
    assert "| Adult completions | 90 |" in document
    assert "18 learning hours to 120 unique adults" in document
    assert "| Learners facing defined access barriers | 72 |" in document


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("partners",), [], "partners"),
        (("rationale",), "", "rationale"),
        (("missing_evidence",), [], "missing_evidence"),
        (("partners", 2, "country"), "Greece", "three organisations"),
        (("partners", 2, "country"), "United States", "recipe supports"),
        (("partners", 0, "years_established_at_deadline"), 1, "coordinator"),
        (("action",), "KA210-ADU", "supports KA220-ADU"),
        (("duration_months",), 37, "12 and 36"),
        (("grant_eur",), 260000, "lump-sum tier"),
        (("estimated_cost_eur",), 250000, "must exceed"),
        (("grant_eur",), True, "integer"),
        (("schema_version",), True, "integer"),
        (("work_packages", 0, "partner_grants", "P1"), 24000, "allocations"),
        (("work_packages", 0, "cost_items", 0, "amount_eur"), 39000, "cost assumptions"),
        (("work_packages", 3, "activities", 0, "end"), 25, "activity dates"),
        (("work_packages", 2, "activities", 1, "end"), 11, "must finish before"),
        (("learners_per_partner",), 5, "subsets"),
        (("risks", 0, "owner"), "unknown", "risk owner"),
    ],
)
def test_incomplete_or_contradictory_briefs_rejected_before_agents_or_outputs(
    tmp_path, brief, monkeypatch, path, value, message
):
    invalid = copy.deepcopy(brief)
    parent = invalid
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value

    def unexpected_agent(*args, **kwargs):
        raise AssertionError("invalid inputs must be rejected before constructing agents")

    monkeypatch.setattr(erasmus, "Agent", unexpected_agent)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=message):
        asyncio.run(erasmus.run(output, brief=invalid))
    assert not output.exists()


def test_no_artificial_lesson_when_first_draft_is_clean(tmp_path):
    report = asyncio.run(erasmus.run(tmp_path / "clean", inject_faults=False))
    assert report["initial_error_count"] == report["final_error_count"] == 0
    assert report["learning_revision"] == 0
    assert report["remaining_evidence_warnings"] == 6
    assert report["submission_ready"] is False


def test_cli_writes_usable_artifacts_and_refuses_to_overwrite_them(tmp_path):
    output = tmp_path / "cli"
    command = [sys.executable, "-m", "examples.erasmus_proposal", "--output-dir", str(output)]
    result = subprocess.run(
        command, text=True, encoding="utf-8", capture_output=True, env=os.environ
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["final_error_count"] == 0
    before = (output / "proposal.md").read_bytes()
    repeated = subprocess.run(
        command, text=True, encoding="utf-8", capture_output=True, env=os.environ
    )
    assert repeated.returncode == 2
    assert "existing artifacts are not overwritten" in repeated.stderr
    assert (output / "proposal.md").read_bytes() == before


@pytest.mark.parametrize("precreate_empty", [False, True])
def test_competing_run_cannot_share_or_replace_winners_artifacts(
    tmp_path, monkeypatch, precreate_empty
):
    output = tmp_path / "contended"
    if precreate_empty:
        output.mkdir()
    original_generate = erasmus.QAgentsGenerator.generate
    planner_calls = 0

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def pause_first_planner(self, task, context):
            nonlocal planner_calls
            if task.task_id == "proposal-plan":
                planner_calls += 1
                if planner_calls == 1:
                    entered.set()
                    await release.wait()
            return await original_generate(self, task, context)

        monkeypatch.setattr(erasmus.QAgentsGenerator, "generate", pause_first_planner)
        winner = asyncio.create_task(erasmus.run(output))
        await asyncio.wait_for(entered.wait(), timeout=5)
        contender = (await asyncio.gather(erasmus.run(output), return_exceptions=True))[0]
        release.set()
        winning_result = (await asyncio.gather(winner, return_exceptions=True))[0]
        return contender, winning_result

    contender, winner = asyncio.run(scenario())
    assert isinstance(contender, ValueError)
    assert planner_calls == 1
    assert winner["internal_checks_passed"] is True
    with SQLiteStore(output / "gym.sqlite3") as store:
        assert len(store.episodes("erasmus-proposal")) == 2
        assert len(store.changes("erasmus-proposal")) == 1
    assert json.loads((output / "workflow.json").read_text())["agent_calls"]["planner"] == 1


@pytest.mark.parametrize("filename", ["proposal.md", "brief.json", "episodes.jsonl", "gym.sqlite3"])
def test_file_created_during_planning_is_never_overwritten(tmp_path, monkeypatch, filename):
    output = tmp_path / "external-file"
    original_generate = erasmus.QAgentsGenerator.generate
    preserved = b"Owned by another process; preserve these exact bytes."

    async def create_file_during_planning(self, task, context):
        if task.task_id == "proposal-plan":
            output.mkdir(parents=True, exist_ok=True)
            (output / filename).write_bytes(preserved)
        return await original_generate(self, task, context)

    monkeypatch.setattr(erasmus.QAgentsGenerator, "generate", create_file_during_planning)
    with pytest.raises(FileExistsError):
        asyncio.run(erasmus.run(output))
    assert (output / filename).read_bytes() == preserved


def test_failed_run_keeps_reservation_and_requires_a_fresh_destination(tmp_path, monkeypatch):
    output = tmp_path / "failed"

    async def fail_planning(self, task, context):
        raise RuntimeError("injected planning failure")

    monkeypatch.setattr(erasmus.QAgentsGenerator, "generate", fail_planning)
    with pytest.raises(RuntimeError, match="injected planning failure"):
        asyncio.run(erasmus.run(output))
    assert {file.name for file in output.iterdir()} == {".run-reservation"}
    with pytest.raises(ValueError, match="existing artifacts are not overwritten"):
        asyncio.run(erasmus.run(output))
