import asyncio
import json

import pytest
from qagents import AgentRuntime, RunInterrupted, SQLiteStore

from examples import background_evaluation as evaluation
from examples import background_worker as reports


def test_queued_jobs_are_persisted_and_cancelled_job_does_no_work(tmp_path):
    path = tmp_path / "reports.sqlite3"
    report = asyncio.run(reports.run(path))
    assert report["accepted_store_status"] == dict.fromkeys(
        ("first", "second", "cancelled"), "running"
    )
    assert report["queued_handle_status"] == {"second": "queued", "cancelled": "queued"}
    assert report["results"] == {
        "first": "completed",
        "second": "completed",
        "cancelled": "cancelled",
    }
    assert report["outputs"] == ["Report: Evidence collected.", "Report: Budget checked."]
    assert report["source_reads"] == ["first", "second"]
    assert report["cancelled_model_requests"] == 0
    assert report["reopened_store_status"] == dict.fromkeys(
        ("first", "second", "cancelled"), "idle"
    )
    assert report["recoverable_after_finish"] == 0

    async def inspect_cancelled_journal():
        async with SQLiteStore(path) as store:
            entries = await store.load(report["session_ids"]["cancelled"])
            assert [entry.kind for entry in entries].count("run_started") == 1
            assert [entry.kind for entry in entries].count("run_finished") == 1
            assert not any(entry.kind.startswith("model_") for entry in entries)
            assert not any(entry.kind.startswith("tool_") for entry in entries)

    asyncio.run(inspect_cancelled_journal())


def test_frozen_evaluation_reuses_completed_score_after_database_reopen(tmp_path):
    path = tmp_path / "evaluation.sqlite3"
    report = asyncio.run(evaluation.run(path))
    assert report["status_after_shutdown"] == "running"
    assert report["lease_released"] is True
    assert report["same_run_after_recovery"] is True
    assert report["calls_before_shutdown"] == ["small-sum"]
    assert report["calls_after_recovery"] == ["large-sum"]
    assert report["scores"] == [
        {"task_id": "small-sum", "answer": "5", "reward": 1.0},
        {"task_id": "large-sum", "answer": "42", "reward": 1.0},
    ]
    assert report["final_status"] == "completed"
    assert report["final_store_status"] == "idle"
    assert report["frozen_context"] is True
    assert report["recoverable_after_finish"] == 0

    async def inspect_recovered_journal():
        async with SQLiteStore(path) as store:
            entries = await store.load(report["session_id"])
            kinds = [entry.kind for entry in entries]
            assert kinds.count("run_started") == 1
            assert kinds.count("run_recovered") == 1
            assert kinds.count("tool_result") == 2
            assert kinds.count("run_finished") == 1

    asyncio.run(inspect_recovered_journal())


def test_second_worker_cannot_recover_an_active_lease(tmp_path):
    async def scenario():
        path = tmp_path / "leased.sqlite3"
        inputs = reports.ReportInputs({"first": "ready"}, release=asyncio.Event())
        async with SQLiteStore(path) as store_a, SQLiteStore(path) as store_b:
            worker_a = AgentRuntime(
                store_a, [reports.report_agent()], deps_factory=lambda session_id: inputs
            )
            worker_b = AgentRuntime(store_b, [reports.report_agent()])
            try:
                handle = await worker_a.start(
                    "background-report", json.dumps({"report_id": "first"}), session_id="leased"
                )
                await asyncio.wait_for(inputs.started.wait(), 5)
                assert (await store_b.status("leased")).lease_owner is not None
                assert worker_b.get("leased") is None  # Handles are local to one runtime.
                assert await worker_b.recover() == []
                inputs.release.set()
                assert (await asyncio.wait_for(handle.wait(), 5)).completed
                assert inputs.calls == ["first"]
            finally:
                inputs.release.set()
                await worker_a.shutdown(timeout_s=1)
                await worker_b.shutdown(timeout_s=1)

    asyncio.run(scenario())


def test_shutdown_preserves_queued_job_before_its_first_model_call(tmp_path):
    async def scenario():
        path = tmp_path / "queued.sqlite3"
        inputs = reports.ReportInputs({"first": "A", "second": "B"}, release=asyncio.Event())
        async with SQLiteStore(path) as store:
            worker = AgentRuntime(
                store,
                [reports.report_agent()],
                max_concurrent=1,
                deps_factory=lambda session_id: inputs,
            )
            try:
                first = await worker.start(
                    "background-report", json.dumps({"report_id": "first"}), session_id="first"
                )
                await asyncio.wait_for(inputs.started.wait(), 5)
                queued = await worker.start(
                    "background-report", json.dumps({"report_id": "second"}), session_id="queued"
                )
                assert queued.status == "queued"
                await worker.shutdown(timeout_s=0)
                for handle in (first, queued):
                    with pytest.raises(RunInterrupted):
                        await handle.wait()
                assert (await store.status("queued")).status == "running"
                assert (await store.status("queued")).lease_owner is None
                assert inputs.calls == ["first"]
            finally:
                inputs.release.set()
                await worker.shutdown(timeout_s=1)
        # Recover only the queued job; the interrupted read is a separate concern.
        fresh = reports.ReportInputs({"second": "B"})
        async with SQLiteStore(path) as store:
            worker = AgentRuntime(
                store,
                [reports.report_agent()],
                session_prefix="queued",
                deps_factory=lambda session_id: fresh,
            )
            try:
                handles = await worker.recover()
                assert len(handles) == 1
                result = await asyncio.wait_for(handles[0].wait(), 5)
                assert result.text == "Report: B"
                assert fresh.calls == ["second"]
                assert (await store.status("first")).status == "running"
            finally:
                await worker.shutdown(timeout_s=1)

    asyncio.run(scenario())


def test_answer_keys_are_excluded_from_the_persisted_model_input():
    prompt = evaluation.evaluation_prompt(evaluation.Context("heldout"))
    payload = json.loads(prompt)
    assert payload["revision"] == 0
    assert set(payload) == {"context", "revision", "tasks"}
    assert all(set(task) == {"task_id", "prompt"} for task in payload["tasks"])
    assert "answers" not in payload

    # A private sentinel answer must affect scoring, never the generated candidate.
    async def scenario():
        inputs = evaluation.EvaluationInputs(
            evaluator=evaluation.ExactMatchEvaluator(dict.fromkeys(evaluation.ANSWERS, "PRIVATE"))
        )
        result = await evaluation.evaluation_agent().run(prompt, deps=inputs)
        assert result.completed
        rows = json.loads(result.text)
        assert [row["answer"] for row in rows] == ["5", "42"]
        assert [row["reward"] for row in rows] == [0.0, 0.0]

    asyncio.run(scenario())


@pytest.mark.parametrize("example", [reports, evaluation])
def test_examples_preserve_previous_batches_when_reusing_database(tmp_path, example):
    path = tmp_path / "repeat.sqlite3"
    first = asyncio.run(example.run(path, namespace="same"))
    second = asyncio.run(example.run(path, namespace="same"))
    first_ids = set(first.get("session_ids", {}).values()) or {first["session_id"]}
    second_ids = set(second.get("session_ids", {}).values()) or {second["session_id"]}
    assert first_ids.isdisjoint(second_ids)

    async def inspect():
        async with SQLiteStore(path) as store:
            saved = {item.session_id for item in await store.list_sessions()}
            assert saved == first_ids | second_ids

    asyncio.run(inspect())
