"""Controller checks plus opt-in native crash trials using the active Python environment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.recovery.protocol import read_json, request_effect, write_json
from benchmarks.recovery.run import (
    Inconclusive,
    classify,
    distribution,
    interpreter_command,
    run_trial,
    stop_owned,
    wait_for,
)
from benchmarks.recovery.service import EffectService, Ledger, PayloadConflict


@pytest.mark.parametrize("deduplicate, expected", [(True, 1), (False, 2)])
def test_ledger_commits_and_deduplicates_by_business_intent(tmp_path, deduplicate, expected):
    ledger = Ledger(tmp_path / "ledger.sqlite3", deduplicate=deduplicate)
    first = ledger.credit("order-1", 10, "first")
    second = ledger.credit("order-1", 10, "recovery")
    state = ledger.snapshot()
    assert state["request_count"] == 2
    assert state["credit_count"] == expected
    assert state["balance"] == expected * 10
    assert first["credit_id"] == 1
    assert second["credit_id"] == expected


def test_conflicting_payload_is_rejected_without_changing_ledger(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3", deduplicate=True)
    ledger.credit("order-1", 10, "first")
    before = ledger.snapshot()
    with pytest.raises(PayloadConflict):
        ledger.credit("order-1", 20, "recovery")
    assert ledger.snapshot() == before
    ledger.credit("order-2", 10, "first")
    assert ledger.snapshot()["credit_count"] == 2


def test_actual_loopback_http_returns_durable_receipt(tmp_path):
    with EffectService(tmp_path, fault="none", deduplicate=True, timeout=2) as service:
        receipt = request_effect(service.url, "order-1", "first", 2)
        replay = request_effect(service.url, "order-1", "recovery", 2)
        assert receipt == replay == {"key": "order-1", "amount": 10, "credit_id": 1}
        assert service.ledger.snapshot()["request_count"] == 2


def observed_case():
    return {
        "fault": "after_commit_before_ack",
        "deduplicate": True,
        "key": "order-1",
        "kill": {
            "pid": 123,
            "reported_worker_pid": 123,
            "alive_before_kill": True,
            "returncode": -9,
        },
        "barrier": {"name": "after_commit_before_ack", "pid": 456},
        "first_result_exists_at_kill": False,
        "ledger_before_recovery": {"credit_count": 1, "request_count": 1, "balance": 10},
        "pre_recovery": {"tool_result_count": 0, "tool_results": []},
        "result": {
            "receipt": {"key": "order-1", "amount": 10, "credit_id": 1},
            "framework_status": "completed",
        },
        "ledger": {"request_count": 2, "credit_count": 1, "balance": 10},
    }


def test_correct_receipt_cannot_hide_missing_kill_or_checkpoint_evidence():
    case = observed_case()
    assert classify(case) == ("passed", [])
    case["kill"]["returncode"] = 0
    assert classify(case)[0] == "inconclusive"
    case["kill"]["returncode"] = -9
    case["pre_recovery"]["tool_result_count"] = 1
    assert classify(case)[0] == "inconclusive"
    case["pre_recovery"]["tool_result_count"] = 0
    case["ledger"]["credit_count"] = 2
    case["ledger"]["balance"] = 20
    assert classify(case)[0] == "failed"


def test_duplicate_control_requires_observed_duplicate():
    case = observed_case()
    case["deduplicate"] = False
    assert classify(case)[0] == "failed"
    case["ledger"].update(credit_count=2, balance=20)
    case["result"]["receipt"]["credit_id"] = 2
    assert classify(case) == ("expected_duplicate_detected", [])


@pytest.mark.parametrize(
    "field", ["kill", "barrier", "result", "ledger", "pre_recovery", "ledger_before_recovery"]
)
@pytest.mark.parametrize("malformed", [None, [], "not-an-object", 42])
def test_malformed_evidence_objects_are_inconclusive(field, malformed):
    case = observed_case()
    case[field] = malformed
    assert classify(case)[0] == "inconclusive"


@pytest.mark.parametrize(
    "path,value",
    [
        (("ledger", "request_count"), None),
        (("ledger", "credit_count"), True),
        (("ledger", "balance"), "10"),
        (("kill", "returncode"), False),
        (("kill", "pid"), True),
        (("barrier", "pid"), []),
        (("pre_recovery", "tool_result_count"), False),
        (("pre_recovery", "tool_results"), {}),
        (("result", "receipt"), []),
        (("result", "receipt", "credit_id"), True),
        (("result", "receipt", "amount"), 10.0),
    ],
)
def test_missing_fields_and_wrong_numeric_types_are_inconclusive(path, value):
    case = observed_case()
    target = case
    for name in path[:-1]:
        target = target[name]
    if value is None:
        target.pop(path[-1])
    else:
        target[path[-1]] = value
    assert classify(case)[0] == "inconclusive"


def test_missing_observation_is_inconclusive():
    assert classify(None)[0] == "inconclusive"
    assert classify({})[0] == "inconclusive"


def test_launcher_pid_mismatch_invalidates_every_fault_kind():
    case = observed_case()
    case["kill"]["reported_worker_pid"] = 999
    assert classify(case)[0] == "inconclusive"


def test_owned_interpreter_pid_is_the_process_executing_python(tmp_path):
    marker_path = tmp_path / "pid.json"
    code = (
        "import json,os,time; "
        f"open({str(marker_path)!r}, 'w').write(json.dumps({{'pid': os.getpid()}})); "
        "time.sleep(30)"
    )
    child = subprocess.Popen(
        interpreter_command(code),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        reported = wait_for(marker_path, child, 5)
        assert reported["pid"] == child.pid
    finally:
        returncode = stop_owned(child)
    assert returncode != 0


def test_missing_barrier_is_inconclusive_even_when_child_exits_successfully(tmp_path):
    child = subprocess.Popen(
        interpreter_command("pass"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        with pytest.raises(Inconclusive, match="before barrier.json"):
            wait_for(tmp_path / "barrier.json", child, 5)
    finally:
        stop_owned(child)
    assert child.returncode == 0


def test_timeout_cleanup_terminates_only_the_owned_child(tmp_path):
    child = subprocess.Popen(
        interpreter_command("import time; time.sleep(30)"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        with pytest.raises(Inconclusive, match="timed out"):
            wait_for(tmp_path / "barrier.json", child, 0.05)
    finally:
        returncode = stop_owned(child)
    assert returncode != 0 and child.poll() is not None


def test_trial_refuses_existing_directory_without_overwrite(tmp_path):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        run_trial(
            tmp_path,
            framework="qagents",
            fault="none",
            deduplicate=True,
            qagents_source=tmp_path,
        )
    assert sentinel.read_text() == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing.txt"]


def test_json_artifacts_refuse_overwrite_and_quantiles_are_explicit(tmp_path):
    path = tmp_path / "raw.json"
    write_json(path, {"measurement": 10})
    with pytest.raises(FileExistsError):
        write_json(path, {"measurement": 20})
    assert read_json(path) == {"measurement": 10}
    assert distribution([]) is None
    assert distribution([9, 1, 5]) == {
        "n": 3,
        "min_ms": 1,
        "median_ms": 5,
        "p95_ms": 9,
        "max_ms": 9,
    }


@pytest.mark.skipif(
    os.environ.get("RUN_RECOVERY_BENCHMARK") != "1",
    reason="native process-crash matrix is opt-in; run in the pinned benchmark environment",
)
@pytest.mark.parametrize("framework", ["qagents", "langgraph"])
@pytest.mark.parametrize(
    "fault,deduplicate",
    [
        ("none", True),
        ("before_effect", True),
        ("after_commit_before_ack", True),
        ("after_checkpoint", True),
        ("after_commit_before_ack", False),
    ],
)
def test_native_recovery_matrix(tmp_path, framework, fault, deduplicate):
    source = Path(os.environ["QAGENT_RECOVERY_SOURCE"])
    observation = run_trial(
        tmp_path / "trial",
        framework=framework,
        fault=fault,
        deduplicate=deduplicate,
        qagents_source=source,
    )
    expected = "passed" if deduplicate else "expected_duplicate_detected"
    assert observation["outcome"] == expected, observation
    assert read_json(tmp_path / "trial" / "observation.json")["outcome"] == expected
    if fault != "none":
        assert observation["kill"]["returncode"] != 0
        assert not observation["first_result_exists_at_kill"]
        assert observation["pre_recovery"]["tool_result_count"] == int(fault == "after_checkpoint")


@pytest.mark.parametrize(
    "field, outcome",
    [
        ("missing_intent", "inconclusive"),
        ("changed_prefix", "failed"),
        ("changed_digest", "failed"),
        ("changed_key", "failed"),
        ("missing_requests", "inconclusive"),
        ("changed_commit", "failed"),
    ],
)
def test_effects_pass_requires_matching_durable_and_remote_evidence(field, outcome):
    case = observed_case()
    first = {"request_digest": "original", "idempotency_key": "q1_stable", "attempt": 1}
    second = {**first, "attempt": 2}
    case["framework"] = "qagents-effects"
    case["pre_recovery"]["effect_intents"] = [first.copy()]
    case["result"]["effect_intents"] = [first, second]
    case["result"]["effect_commits"] = [{"receipt": case["result"]["receipt"].copy()}]
    requests = [{"idempotency_key": first["idempotency_key"]} for _ in range(2)]
    case["ledger"]["requests"] = requests
    assert classify(case) == ("passed", [])

    if field == "missing_intent":
        case["pre_recovery"]["effect_intents"] = []
    elif field == "changed_prefix":
        case["pre_recovery"]["effect_intents"][0]["attempt"] = 9
    elif field == "changed_digest":
        second["request_digest"] = "changed"
    elif field == "changed_key":
        requests[1]["idempotency_key"] = "q1_different"
    elif field == "missing_requests":
        requests.pop()
    elif field == "changed_commit":
        case["result"]["effect_commits"][0]["receipt"]["credit_id"] = 2
    assert classify(case)[0] == outcome
