# Background jobs with QAgents

These two offline examples use QAgents' existing `AgentRuntime` and a SQLite
session journal. They run real agent loops with deterministic `FunctionModel`
responses. No model provider, payment, telephone call, or external write is used.

Run from the repository root after installing QAgents and QAgent Gym:

```sh
python -m examples.background_worker --db reports.sqlite3
python -m examples.background_evaluation --db evaluation.sqlite3
python -m pytest tests/test_background_jobs.py -q
```

Omit `--db` for a temporary demonstration. Repeating a command creates a separate
batch in the same database; it does not delete earlier sessions. The database
contains prompts and tool results, so use synthetic data when sharing it.

## Queue reports and cancel unwanted work

[`examples/background_worker.py`](https://github.com/mpvasilis/qagent-gym/blob/main/examples/background_worker.py) starts three
report jobs with `max_concurrent=1`. The first job waits at a deterministic
read-only fixture boundary, making the other two jobs visibly queued. The example
cancels the third job before it starts and then lets the first two finish.

The output verifies:

- All three jobs are journaled before `start()` returns, including queued jobs.
- The cancelled job makes zero model requests and never reads a report source.
- A new SQLite connection can read all three finished sessions.
- Recovery does not restart completed or cancelled work.

`deps_factory` supplies worker-local clients or fixture data. Those Python objects
are not serialized into the session. Rebuild them in every worker that may
recover a job, and retain the registered agent name.

## Resume a frozen evaluation after a worker restart

[`examples/background_evaluation.py`](https://github.com/mpvasilis/qagent-gym/blob/main/examples/background_evaluation.py) runs a
two-case arithmetic evaluation with Gym's `Task`, `Context`, and
`ExactMatchEvaluator`. The complete frozen task input goes into the durable agent
prompt; trusted answer keys stay in worker dependencies. Scoring occurs only after
the scripted generator has produced its candidate answer.

The worker stops after the first score is committed and before the second scoring
tool starts. The example closes SQLite, constructs a fresh connection, agent,
dependency factory, and runtime, and calls `recover()`. The recovered job keeps its
original run ID, reuses the first recorded score, and executes only the second
scoring tool. Both rewards are `1.0`, and no playbook or memory is changed.

This is a deterministic worker-stop/reopen test in one Python process, not an
operating-system kill or power-loss test. It uses `shutdown(timeout_s=0)` to
interrupt the intentionally waiting hook and release the lease immediately.

The example's durable result is the QAgents journal. It does not invoke
`Gym.evaluate()` or `Gym.train()` as one large retryable tool, and it does not write
Gym episode rows. Those methods can write multiple records; wrapping the whole
method does not make its writes atomic or idempotent. If publishing evaluation
episodes separately, use stable IDs and reconcile the two stores explicitly.

## Use the public runtime in a worker

```python
import asyncio

from qagents import AgentRuntime, SQLiteStore

from examples.background_worker import ReportInputs, report_agent


async def serve():
    async with SQLiteStore("jobs.sqlite3") as store:
        runtime = AgentRuntime(
            store,
            [report_agent()],
            max_concurrent=4,
            session_prefix="reports:",
            deps_factory=lambda session_id: ReportInputs({"first": "Evidence collected."}),
        )
        try:
            # Ingress code calls runtime.start(..., session_id="reports:unique-job-id").
            # This loop recovers previously accepted jobs whose leases are available.
            await runtime.run_forever(poll_interval_s=2)
        finally:
            await runtime.shutdown(timeout_s=10)


if __name__ == "__main__":
    asyncio.run(serve())
```

Run this worker in a supervised process. An application still needs its own
authenticated ingress, job ownership checks, and shutdown signal handling. For
several hosts, use QAgents' optional `PostgresStore` instead of sharing a SQLite
file over a network filesystem. These examples test SQLite only.

| Need | Public API and meaning |
| --- | --- |
| Accept work | `await runtime.start(agent_name, prompt, session_id=...)` journals acceptance before returning a handle. |
| Wait for a result | `await handle.wait()` returns a `RunResult`, or raises the job error / `RunInterrupted`. |
| Cancel work permanently | `handle.cancel(reason)` is synchronous; keep driving the runtime and await completion. The result becomes `cancelled`. |
| Stop a worker for recovery | `await runtime.shutdown(...)` leaves interrupted runs open and releases their leases. |
| Recover work | `await runtime.recover()` takes free or expired leases for supported running/failed sessions. Paused approvals require an explicit `resume()`. |
| Read status from another process | `await store.status(session_id)` or `await store.list_sessions(...)`; a different runtime's `get()` has no access to the old Python handle. |

A handle reports `queued`, `running`, `finished`, `failed`, or `interrupted`.
Persisted session status reports `idle`, `running`, `paused`, or `failed`.
`idle` means no open run: inspect the journal/result to distinguish completion
from cancellation. Queued jobs already appear as `running` in persisted status.

## Recovery boundaries

The tests cover durable acceptance, queued cancellation, lease exclusion between
two SQLite connections, queued-job recovery, recorded-result reuse, and preserved
earlier batches. They do not establish a single-application guarantee across services.

A recorded tool result is reused. An interrupted tool without a recorded result
may run again when declared `read_only` or `idempotent`; an ordinary external write
has an unknown outcome and needs reconciliation. An in-flight model request lost
with its process may be sent again and billed again. Leases protect journal
ownership, not a bank, email provider, or other external destination.

The concurrency queue and handles live in each worker, while accepted runs live
in the journal. This is not a cron scheduler, priority queue, dead-letter service,
or distributed cancellation endpoint. Do not attach those guarantees to these
examples or to the unfinished Effects work.
