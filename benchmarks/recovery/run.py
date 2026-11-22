"""Run matched local recovery trials and retain every observation.

python -m benchmarks.recovery.run --qagents-src path/to/qagents/src \
    --output-dir .work/recovery-run --repeats 5 --include-negative-control
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import site
import sqlite3
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from .protocol import (
    AMOUNT,
    FAULTS,
    FRAMEWORKS,
    clean_environment,
    read_json,
    source_hash,
    write_json,
)
from .service import EffectService


class Inconclusive(RuntimeError):
    """The experiment did not produce sufficient evidence to judge correctness."""


def interpreter_command(code: str) -> list[str]:
    """Bypass the Windows venv redirector so Popen owns the actual Python worker."""
    executable = getattr(sys, "_base_executable", None) or sys.executable
    return [executable, "-I", "-S", "-B", "-u", "-c", code]


def worker_command(config: Path, source: Path) -> list[str]:
    # -I/-S exclude user/global site setup. Add only this environment's installed
    # dependencies and the two explicit source roots; -B also applies with -I.
    root = Path(__file__).resolve().parents[2]
    dependencies = [path for path in site.getsitepackages() if Path(path).is_dir()]
    code = (
        "import runpy,site,sys; "
        f"[site.addsitedir(path) for path in {dependencies!r}]; "
        f"sys.path[:0] = {[str(root), str(source)]!r}; "
        f"sys.argv = ['benchmarks.recovery.worker', {str(config)!r}]; "
        "runpy.run_module('benchmarks.recovery.worker', run_name='__main__')"
    )
    return interpreter_command(code)


def wait_for(path: Path, child: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                result = read_json(path)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # The producer may still be writing this exclusive artifact.
        if child.poll() is not None:
            raise Inconclusive(f"worker exited with {child.returncode} before {path.name}")
        time.sleep(0.005)
    raise Inconclusive(f"timed out waiting for {path.name}")


def stop_owned(child: subprocess.Popen) -> int:
    """Terminate only a child we started; no PID search or process-tree wildcard."""
    if child.poll() is None:
        child.kill()
    return child.wait(timeout=10)


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(values),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max_ms": ordered[-1],
    }


def classify(observation: dict) -> tuple[str, list[str]]:
    """Separate missing evidence from a fully observed correctness failure."""
    if not isinstance(observation, dict):
        return "inconclusive", ["observation is not an object"]
    missing = []

    def object_field(parent: dict, name: str) -> dict:
        value = parent.get(name)
        if not isinstance(value, dict):
            missing.append(f"missing or malformed {name} object")
            return {}
        return value

    def integer_field(parent: dict, name: str, *, minimum: int | None = 0) -> None:
        value = parent.get(name)
        if type(value) is not int or (minimum is not None and value < minimum):
            missing.append(f"missing or malformed {name} integer")

    def ledger_fields(value: dict) -> None:
        for name in ("request_count", "credit_count", "balance"):
            integer_field(value, name)

    fault = observation.get("fault")
    if fault not in FAULTS or type(observation.get("deduplicate")) is not bool:
        missing.append("missing or malformed fault/deduplication configuration")
    if not isinstance(observation.get("key"), str) or not observation["key"]:
        missing.append("missing or malformed operation key")
    result = object_field(observation, "result")
    receipt = object_field(result, "receipt")
    if not isinstance(receipt.get("key"), str):
        missing.append("missing or malformed receipt key")
    integer_field(receipt, "amount", minimum=1)
    integer_field(receipt, "credit_id", minimum=1)
    if not isinstance(result.get("framework_status"), str):
        missing.append("missing or malformed framework status")
    ledger = object_field(observation, "ledger")
    ledger_fields(ledger)
    if fault != "none":
        kill = object_field(observation, "kill")
        integer_field(kill, "pid", minimum=1)
        integer_field(kill, "returncode", minimum=None)
        integer_field(kill, "reported_worker_pid", minimum=1)
        if kill.get("reported_worker_pid") != kill.get("pid"):
            missing.append("owned process PID differs from the executing Python worker")
        if kill.get("alive_before_kill") is not True or kill.get("returncode") in (None, 0):
            missing.append("real worker termination was not observed")
        barrier = object_field(observation, "barrier")
        integer_field(barrier, "pid", minimum=1)
        if barrier.get("name") != fault:
            missing.append("requested barrier was not observed")
        if fault in ("before_effect", "after_checkpoint") and barrier.get("pid") != kill.get("pid"):
            missing.append("fault barrier did not originate from the killed worker")
        if observation.get("first_result_exists_at_kill") is not False:
            missing.append("first worker already published a completed result")
        previous = object_field(observation, "pre_recovery")
        integer_field(previous, "tool_result_count")
        if not isinstance(previous.get("tool_results"), list) or len(
            previous["tool_results"]
        ) != previous.get("tool_result_count"):
            missing.append("malformed persisted tool results")
        before = object_field(observation, "ledger_before_recovery")
        ledger_fields(before)
        expected_before = 0 if fault == "before_effect" else 1
        if before.get("credit_count") != expected_before:
            missing.append("service ledger does not establish the requested fault boundary")
        if fault == "after_checkpoint" and previous.get("tool_result_count") != 1:
            missing.append("completed effect was not present in the persisted checkpoint")
        if fault == "after_commit_before_ack" and previous.get("tool_result_count") != 0:
            missing.append("withheld response was already recorded by the framework")
    if missing:
        return "inconclusive", missing
    if observation.get("framework") == "qagents-effects":
        intents = result.get("effect_intents")
        commits = result.get("effect_commits")
        requests = ledger.get("requests")
        if (
            not isinstance(intents, list)
            or not isinstance(commits, list)
            or not isinstance(requests, list)
            or not intents
            or not commits
            or not all(isinstance(item, dict) for item in [*intents, *commits, *requests])
        ):
            return "inconclusive", ["missing durable Effects or service request evidence"]
        if fault != "none":
            persisted = previous.get("effect_intents")
            if not isinstance(persisted, list) or len(persisted) != 1:
                return "inconclusive", ["expected one durable effect intent before the kill"]
            if persisted[0] != intents[0]:
                return "failed", ["the recovered journal changed the original effect intent"]
        expected_intents = 2 if fault in ("before_effect", "after_commit_before_ack") else 1
        if len(intents) != expected_intents or len(commits) != 1:
            return "failed", ["unexpected durable effect intent or commit count"]
        if len(requests) != ledger["request_count"] or not requests:
            return "inconclusive", ["service request records do not establish the request count"]
        digest = intents[0].get("request_digest")
        if not isinstance(digest, str) or not digest:
            return "inconclusive", ["missing original request digest"]
        if any(intent.get("request_digest") != digest for intent in intents):
            return "failed", ["the recovery tried a different effect request"]
        if commits[0].get("receipt") != receipt:
            return "failed", ["the durable commit differs from the reported receipt"]
        keys = [item.get("idempotency_key") for item in [*intents, *requests]]
        if not all(isinstance(token, str) and token.startswith("q1_") for token in keys):
            return "failed", ["the service did not receive the Effects attempt key"]
        if len(set(keys)) != 1:
            return "failed", ["the service and durable attempts disagree on the retry key"]
    expected_requests = 2 if fault == "after_commit_before_ack" else 1
    negative = fault == "after_commit_before_ack" and not observation["deduplicate"]
    expected_credits = 2 if negative else 1
    errors = []
    if ledger["request_count"] != expected_requests:
        errors.append(f"expected {expected_requests} service requests")
    if ledger["credit_count"] != expected_credits or ledger["balance"] != expected_credits * AMOUNT:
        errors.append(f"expected {expected_credits} credits totaling {expected_credits * AMOUNT}")
    if receipt != {
        "key": observation["key"],
        "amount": AMOUNT,
        "credit_id": expected_credits,
    }:
        errors.append("final receipt does not match the committed service outcome")
    if result.get("framework_status") != "completed":
        errors.append("framework did not report completion")
    if errors:
        return "failed", errors
    return ("expected_duplicate_detected" if negative else "passed"), []


def run_trial(
    directory: Path,
    *,
    framework: str,
    fault: str,
    deduplicate: bool,
    qagents_source: Path,
    timeout: float = 25.0,
    lease_ttl: float = 0.75,
) -> dict:
    if framework not in FRAMEWORKS or fault not in FAULTS:
        raise ValueError("unknown framework or fault")
    if type(deduplicate) is not bool or not 0 < lease_ttl < timeout or timeout > 120:
        raise ValueError("invalid trial configuration")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / "temporary"
    temporary.mkdir()
    key = "credit-intent-1"
    observation = {
        "framework": framework,
        "fault": fault,
        "deduplicate": deduplicate,
        "key": key,
        "case": directory.name,
        "timings": {},
    }
    children = []
    logs = []

    def launch(phase: str, service_url: str) -> tuple[subprocess.Popen, int]:
        config = {
            "framework": framework,
            "fault": fault,
            "directory": str(directory),
            "service_url": service_url,
            "key": key,
            "phase": phase,
            "timeout": timeout,
            "lease_ttl": lease_ttl,
        }
        config_path = directory / f"{phase}.config.json"
        write_json(config_path, config)
        out = (directory / f"{phase}.stdout.log").open("xb")
        err = (directory / f"{phase}.stderr.log").open("xb")
        logs.extend((out, err))
        launched = time.perf_counter_ns()
        child = subprocess.Popen(
            worker_command(config_path, qagents_source),
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            env=clean_environment(qagents_source, temporary),
            cwd=Path(__file__).resolve().parents[2],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        children.append(child)
        return child, launched

    with EffectService(directory, fault=fault, deduplicate=deduplicate, timeout=timeout) as service:
        try:
            first, launched = launch("first", service.url)
            ready = wait_for(directory / "first.ready.json", first, timeout)
            if type(ready.get("pid")) is not int or ready["pid"] != first.pid:
                raise Inconclusive("launched process is not the executing Python worker")
            observation["timings"]["first_launch_to_ready_ms"] = (
                ready["monotonic_ns"] - launched
            ) / 1e6
            if fault == "none":
                result = wait_for(directory / "first.result.json", first, timeout)
                if first.wait(timeout=10) != 0:
                    raise Inconclusive("worker failed after publishing its result")
                observation["timings"]["complete_launch_to_result_ms"] = (
                    result["monotonic_ns"] - launched
                ) / 1e6
            else:
                barrier = wait_for(directory / "barrier.json", first, timeout)
                observation["barrier"] = barrier
                observation["timings"]["ready_to_fault_ms"] = (
                    barrier["monotonic_ns"] - ready["monotonic_ns"]
                ) / 1e6
                alive = first.poll() is None
                observation["first_result_exists_at_kill"] = (
                    directory / "first.result.json"
                ).exists()
                killed_at = time.perf_counter_ns()
                returncode = stop_owned(first)
                observation["kill"] = {
                    "pid": first.pid,
                    "reported_worker_pid": ready["pid"],
                    "alive_before_kill": alive,
                    "returncode": returncode,
                    "monotonic_ns": killed_at,
                }
                observation["ledger_before_recovery"] = service.ledger.snapshot()
                service.release.set()
                recovered, recovery_launched = launch("recovery", service.url)
                recovery_ready = wait_for(directory / "recovery.ready.json", recovered, timeout)
                if (
                    type(recovery_ready.get("pid")) is not int
                    or recovery_ready["pid"] != recovered.pid
                ):
                    raise Inconclusive("recovery process is not the executing Python worker")
                result = wait_for(directory / "recovery.result.json", recovered, timeout)
                if recovered.wait(timeout=10) != 0:
                    raise Inconclusive("recovery worker failed after publishing its result")
                observation["pre_recovery"] = read_json(directory / "pre_recovery.json")
                observation["timings"].update(
                    recovery_launch_to_ready_ms=(recovery_ready["monotonic_ns"] - recovery_launched)
                    / 1e6,
                    recovery_launch_to_result_ms=(result["monotonic_ns"] - recovery_launched) / 1e6,
                    kill_to_result_ms=(result["monotonic_ns"] - killed_at) / 1e6,
                )
            observation["result"] = result
            observation["timings"].update(
                worker_work_ms=result["work_ms"],
                lease_acquisition_wait_ms=result["lease_acquisition_wait_ms"],
            )
        except Exception as error:
            observation["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            service.release.set()
            for child in children:
                stop_owned(child)
            for log in logs:
                log.close()
            observation["ledger"] = service.ledger.snapshot()
    outcome, reasons = classify(observation)
    observation.update(outcome=outcome, reasons=reasons)
    write_json(directory / "observation.json", observation)
    return observation


def metadata(source: Path) -> dict:
    packages = {}
    for name in (
        "langgraph",
        "langgraph-checkpoint",
        "langgraph-checkpoint-sqlite",
        "langchain-core",
        "httpx",
        "pydantic",
        "matplotlib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    project = source.parent / "pyproject.toml"
    qagents_version = (
        tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"]
        if project.is_file()
        else "unavailable; see source hash"
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "sqlite": sqlite3.sqlite_version,
        "packages": packages,
        "installed_distributions": {
            item.metadata["Name"]: item.version for item in importlib.metadata.distributions()
        },
        "qagents_version": qagents_version,
        "qagents_source_sha256": source_hash(source),
        "sqlite_mode": "WAL / synchronous=FULL",
        "langgraph_durability": "sync",
        "timing_clock": "time.perf_counter_ns",
        "fault_scope": "worker process terminated; controller/service/host remain alive",
    }


def summarize(observations: list[dict]) -> dict:
    groups = {}
    for record in observations:
        name = f"{record['framework']}/{record['fault']}/dedupe={record['deduplicate']}"
        group = groups.setdefault(name, {"outcomes": {}, "timings": {}})
        outcome = record["outcome"]
        group["outcomes"][outcome] = group["outcomes"].get(outcome, 0) + 1
        if outcome in ("passed", "expected_duplicate_detected"):
            for key, value in record["timings"].items():
                group["timings"].setdefault(key, []).append(value)
    for group in groups.values():
        group["timings"] = {key: distribution(values) for key, values in group["timings"].items()}
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qagents-src", type=Path, required=True)
    parser.add_argument("--label", default="QAgents baseline; inspect source hash")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--frameworks", nargs="+", choices=FRAMEWORKS, default=["qagents", "langgraph"]
    )
    parser.add_argument("--faults", nargs="+", choices=FAULTS, default=list(FAULTS))
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--lease-ttl", type=float, default=0.75)
    parser.add_argument("--include-negative-control", action="store_true")
    args = parser.parse_args()
    source = args.qagents_src.resolve()
    if not (source / "qagents" / "__init__.py").is_file():
        parser.error("--qagents-src must contain qagents/__init__.py")
    if not 1 <= args.repeats <= 100 or not 0 < args.lease_ttl < args.timeout <= 120:
        parser.error("require 1..100 repeats and 0 < lease-ttl < timeout <= 120")
    if "qagents-effects" in args.frameworks and args.lease_ttl * 0.999 < 6:
        parser.error("the Effects profile needs a lease TTL of at least 6.01 seconds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "label": args.label,
        "repeats": args.repeats,
        "frameworks": args.frameworks,
        "faults": args.faults,
        "negative_control": args.include_negative_control,
        "timeout_s": args.timeout,
        "lease_ttl_s": args.lease_ttl,
    }
    environment = metadata(source)
    write_json(output / "metadata.json", {**environment, "configuration": config})
    records = []
    cases = [(fault, True) for fault in args.faults]
    if args.include_negative_control:
        cases.append(("after_commit_before_ack", False))
    for repeat in range(args.repeats):
        frameworks = args.frameworks if repeat % 2 == 0 else list(reversed(args.frameworks))
        for fault, deduplicate in cases:
            for framework in frameworks:
                case = f"{repeat:02d}-{framework}-{fault}-dedupe{int(deduplicate)}"
                record = run_trial(
                    output / case,
                    framework=framework,
                    fault=fault,
                    deduplicate=deduplicate,
                    qagents_source=source,
                    timeout=args.timeout,
                    lease_ttl=args.lease_ttl,
                )
                records.append(record)
                print(json.dumps({"case": case, "outcome": record["outcome"]}), flush=True)
    write_json(output / "raw.json", records)
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    stable_source = source_hash(source) == environment["qagents_source_sha256"]
    write_json(output / "source_integrity.json", {"unchanged_during_run": stable_source})
    successful = all(
        record["outcome"] in ("passed", "expected_duplicate_detected") for record in records
    )
    successful = successful and stable_source
    print(json.dumps({"trials": len(records), "all_conclusive_expected": successful}), flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
