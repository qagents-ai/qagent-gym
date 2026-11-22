"""Rehearse KA220 proposal recovery with three offline agent workers.

Run from QAgent Gym with a QAgents build that includes persistent TeamMailbox.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import time
from contextlib import asynccontextmanager, closing
from pathlib import Path

from qagents import Agent, Hooks, Session, SQLiteStore, TeamMailbox
from qagents.models.testing import ScriptedModel, call

from examples.erasmus_proposal import (
    DEFAULT_BRIEF,
    compute_plan,
    render_proposal,
    review_proposal,
    validate_brief,
)

MEMBERS = ["coordinator", "writer", "reviewer"]
DATABASES = ("mailbox.sqlite3", *(f"{name}.sqlite3" for name in MEMBERS))
LEASE_S = 5


def mailbox(directory):
    return TeamMailbox(MEMBERS, db_path=directory / DATABASES[0], max_text_chars=64_000)


def send(recipient, text):
    return call("send_message", recipient=recipient, text=text)


def receive():
    return call("receive_messages", wait_s=20)


def read_mail(request, sender):
    item = request.messages[-1].tool_results[-1]
    if item.is_error:
        raise RuntimeError(item.content)
    (message,) = json.loads(item.content)
    if message["sender"] != sender:
        raise RuntimeError(f"Expected a message from {sender}.")
    return message["text"]


class InterruptAfterDraft(Hooks):
    async def after_tool(self, ctx, call, result):
        if call.name == "send_message" and not result.is_error:
            print(json.dumps({"pid": os.getpid(), "sent": json.loads(result.content)}), flush=True)
            # The inbox transaction committed. The agent journal still lacks the result.
            await asyncio.Event().wait()


async def run_worker(directory, mode):
    brief = validate_brief(json.loads(DEFAULT_BRIEF.read_text(encoding="utf-8")))
    plan = compute_plan(brief)
    role = {
        "assign": "coordinator",
        "draft": "writer",
        "resume": "writer",
        "review": "reviewer",
        "release": "coordinator",
    }[mode]

    def draft(request):
        if read_mail(request, "coordinator") != "Draft BRIDGE":
            raise RuntimeError("Missing the BRIDGE assignment.")
        return send("reviewer", render_proposal(brief, plan, inject_faults=True))

    def review(request):
        first = review_proposal(read_mail(request, "writer"), brief)
        if not first["errors"]:
            raise RuntimeError("The initial draft should trigger the validation checks.")
        return send("writer", json.dumps(first))

    def revise(request):
        first = json.loads(read_mail(request, "reviewer"))
        if not first["errors"]:
            raise RuntimeError("The writer expected actionable review feedback.")
        return send("reviewer", render_proposal(brief, plan))

    def final_review(request):
        report = review_proposal(read_mail(request, "writer"), brief)
        if not report["internal_checks_passed"]:
            raise RuntimeError("The revised draft failed the proposal checks.")
        return send("coordinator", json.dumps(report))

    def release(request):
        report = json.loads(read_mail(request, "reviewer"))
        if not report["internal_checks_passed"]:
            raise RuntimeError("The coordinator received no successful review.")
        return "Fictional BRIDGE draft cleared the internal checks."

    script = {
        "assign": [send("writer", "Draft BRIDGE"), "Assigned"],
        "draft": [receive(), draft],
        "resume": [receive(), revise, "Revised"],
        "review": [receive(), review, receive(), final_review, "Reviewed"],
        "release": [receive(), release],
    }[mode]
    agent = Agent(
        ScriptedModel(script),
        name=role,
        tools=mailbox(directory).endpoint(role).tools(),
        lease_ttl_s=LEASE_S,
        hooks=[InterruptAfterDraft()] if mode == "draft" else [],
    )
    async with SQLiteStore(directory / f"{role}.sqlite3") as store:
        session = Session(store, id=role)
        if mode == "resume":
            result = await agent.resume(session)
        else:
            result = await agent.run(f"Run the BRIDGE {mode} stage.", session=session)
    if not result.completed:
        raise RuntimeError(f"{role} stopped with {result.status}.")
    print(json.dumps({"pid": os.getpid(), "mode": mode, "result": result.text}), flush=True)


@asynccontextmanager
async def worker(directory, mode):
    interpreter = getattr(sys, "_base_executable", sys.executable)
    bootstrap = (
        "import runpy,sys; sys.path[:0]=sys.argv[1].split(chr(31)); "
        "sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    child = await asyncio.create_subprocess_exec(
        interpreter,
        "-c",
        bootstrap,
        chr(31).join(sys.path),
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        str(directory),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        yield child
    finally:
        if child.returncode is None:
            child.kill()
        await asyncio.wait_for(child.communicate(), 10)


async def complete(directory, mode):
    async with worker(directory, mode) as child:
        output, error = await asyncio.wait_for(child.communicate(), 35)
        if child.returncode:
            raise RuntimeError(f"{mode} failed: {error.decode(errors='replace')}")
        result = json.loads(output)
        if result["pid"] != child.pid or result["mode"] != mode:
            raise RuntimeError("The completion report did not identify the owned worker.")
        return result


def snapshot(source, destination):
    """Copy a quiesced set of databases and check each copy before using it."""
    destination.mkdir()
    for name in DATABASES:
        with (
            closing(sqlite3.connect((source / name).as_uri() + "?mode=ro", uri=True)) as original,
            closing(sqlite3.connect(destination / name)) as copied,
        ):
            original.backup(copied)
            if copied.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError(f"SQLite integrity check failed for {name}.")


async def read_results(directory, role, tool):
    async with SQLiteStore(directory / f"{role}.sqlite3") as store:
        entries = await store.load(role)
    return [
        json.loads(entry.data["result"]["content"])
        for entry in entries
        if entry.kind == "tool_result"
        and entry.data["result"]["name"] == tool
        and not entry.data["result"]["is_error"]
    ]


async def recover(directory, earliest_resume, draft):
    started = time.perf_counter()
    async with asyncio.TaskGroup() as tasks:
        reviewer = tasks.create_task(complete(directory, "review"))
        await asyncio.sleep(max(0, earliest_resume - time.time()))
        writer = tasks.create_task(complete(directory, "resume"))
    finished = await complete(directory, "release")
    reviewed = await read_results(directory, "reviewer", "receive_messages")
    sent = await read_results(directory, "writer", "send_message")
    reviewer_sends = await read_results(directory, "reviewer", "send_message")
    feedback, report = [json.loads(message["text"]) for message in reviewer_sends]
    (delivered,) = await read_results(directory, "coordinator", "receive_messages")
    if reviewed != [[draft], [sent[1]]] or sent[0] != draft:
        raise RuntimeError("The recovered draft was missing, reordered or duplicated.")
    sequence = [
        message["sequence"] for message in (sent[0], reviewer_sends[0], sent[1], delivered[0])
    ]
    if sequence != [2, 3, 4, 5] or delivered != [reviewer_sends[1]]:
        raise RuntimeError("The coordinator did not receive the final review.")
    inbox = mailbox(directory)
    if any(await asyncio.gather(*(inbox.endpoint(name).receive() for name in MEMBERS))):
        raise RuntimeError("Messages remained queued after proposal delivery.")
    document = sent[1]["text"]
    with (directory / "proposal.md").open("x", encoding="utf-8", newline="") as output:
        output.write(document)
    return {
        "worker_pids": [writer.result()["pid"], reviewer.result()["pid"], finished["pid"]],
        "recovery_phase_s": round(time.perf_counter() - started, 3),
        "message_sequences": [1, *sequence],
        "initial_errors": len(feedback["errors"]),
        "final_errors": len(report["errors"]),
        "evidence_warnings": len(report["warnings"]),
        "checks": report["checks_run"],
        "proposal_sha256": hashlib.sha256(document.encode()).hexdigest(),
    }


async def pilot(output):
    output.mkdir(parents=True)
    live = output / "live"
    live.mkdir()
    mailbox(live)
    for role in MEMBERS:
        async with SQLiteStore(live / f"{role}.sqlite3") as store:
            await store.load(role)
    assigned = await complete(live, "assign")
    async with worker(live, "draft") as child:
        line = await asyncio.wait_for(child.stdout.readline(), 20)
        if not line:
            raise RuntimeError((await child.stderr.read()).decode(errors="replace"))
        boundary = json.loads(line)
        if boundary["pid"] != child.pid or child.returncode is not None:
            raise RuntimeError("Cannot verify the writer process at the crash boundary.")
        child.kill()
        exit_code = await asyncio.wait_for(child.wait(), 10)
        if exit_code == 0:
            raise RuntimeError("The writer exited cleanly before the interruption.")
    with closing(sqlite3.connect(live / DATABASES[0])) as conn:
        queued = conn.execute(
            "SELECT sequence, body FROM team_messages WHERE recipient = ?", ("reviewer",)
        ).fetchall()
    if queued != [(boundary["sent"]["sequence"], boundary["sent"]["text"])]:
        raise RuntimeError("The initial draft was not retained after termination.")
    async with SQLiteStore(live / "writer.sqlite3") as store:
        entries = await store.load("writer")
        started = any(
            entry.kind == "tool_started" and entry.data["name"] == "send_message"
            for entry in entries
        )
        recorded = any(
            entry.kind == "tool_result" and entry.data["result"]["name"] == "send_message"
            for entry in entries
        )
        if not started or recorded:
            raise RuntimeError("The writer journal was not at the expected send boundary.")
        status = await store.status("writer")
        earliest_resume = (status.lease_expires_at or 0) + store.takeover_grace_s + 0.1
    backup, restored = output / "backup", output / "restored"
    await asyncio.to_thread(snapshot, live, backup)
    restarted = await recover(live, earliest_resume, boundary["sent"])
    await asyncio.to_thread(snapshot, backup, restored)
    recovered = await recover(restored, earliest_resume, boundary["sent"])
    if restarted["proposal_sha256"] != recovered["proposal_sha256"]:
        raise RuntimeError("The recovered proposal differed from the restarted proposal.")
    report = {
        "action": "KA220-ADU",
        "scenario": "Fictional BRIDGE draft using offline scripted policies",
        "assignment_pid": assigned["pid"],
        "terminated_writer_pid": boundary["pid"],
        "terminated_writer_exit": exit_code,
        "backup_files": list(DATABASES),
        "restart": restarted,
        "restore": recovered,
        "matching_proposals": True,
        "submission_ready": False,
    }
    with (output / "pilot-report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["assign", "draft", "resume", "review", "release"])
    parser.add_argument("directory", type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    if args.worker:
        asyncio.run(run_worker(args.directory, args.worker))
    else:
        asyncio.run(pilot(args.directory))
