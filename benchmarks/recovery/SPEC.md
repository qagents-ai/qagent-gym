# Recovery benchmark contract

Date: 2026-11-13. This benchmark compares declared recovery behavior, not model quality.

## Workload and boundaries

- One deterministic plan, one remote synthetic credit of 10 TEST units, one final response.
- A controller-owned HTTP service binds only to loopback and owns a separate SQLite ledger.
- The durable business operation key and payload are identical across framework implementations.
- QAgents uses its public AgentRuntime, SQLiteStore and deterministic FunctionModel.
- LangGraph uses native StateGraph, SqliteSaver and synchronous checkpoint durability.
- Both framework stores and the service ledger use SQLite WAL with synchronous=FULL.
- The host process survives. The experiment tests worker-process termination, not power loss,
  filesystem corruption, multi-host failover, or a real external provider's guarantees.
- Runtime-specific planning and persistence overhead remains visible and is documented.

## Fault protocol

Each trial has a fresh exclusive directory, separate framework database and service ledger.
The first worker blocks at a requested barrier; the controller checks the barrier and kills
that exact child process using its Popen handle. A second child recovers the same session.
No process-name matching, global process termination, source mutation or paid model is used.

| Fault | Durable witness before kill | Expected replay with service deduplication |
| --- | --- | --- |
| before_effect | Worker reached the effect boundary; service ledger has no credit | One credit |
| after_commit_before_ack | Service committed the credit and withheld its HTTP response | Repeated request, one credit |
| after_checkpoint | Framework recorded the tool/node result; finalization has started | No repeated effect request, one credit |

The after-checkpoint barrier is reached only after QAgents supplies a persisted tool result
to the next model request, or LangGraph starts the next node under synchronous durability.
After the kill, the controller also inspects the persisted framework state.

## Controls and honest outcomes

- A no-fault control checks the complete workload.
- A deduplication-disabled negative control intentionally violates the declared idempotent
  service contract. The commit-before-ack case must expose duplicate credits in both runtimes.
  This is an unsafe configuration control, not a claim that properly declared unsafe tools
  should automatically retry.
- A fault trial is inconclusive if the requested barrier was not reached, termination was
  not observed, a durable witness is missing, recovery timed out, or a protocol artifact is
  malformed. It must never be counted as passed.
- A complete observation inconsistent with its expected request/credit/result counts fails.
- A benchmark report separates passed, failed, inconclusive and expected-negative-control
  results. It preserves raw per-trial JSON, stdout/stderr, events and both databases.

## Timing and reproducibility

- Record cold subprocess launch-to-ready separately from execution and recovery duration.
- Recovery includes imports, reopening storage and actual framework recovery. QAgents lease
  expiry waiting is reported separately and included in total time since kill.
- Run repeated trials, alternating framework order; retain individual observations and
  report count, median, minimum, maximum and nearest-rank p95. Small samples are descriptive.
- Record Python, OS, CPU, SQLite, installed SDK versions, QAgents version/source hash,
  run configuration, persistence mode and fault/lease/polling timing parameters.
- Do not rank speeds when persistence modes, correctness or recovery policies differ.

## Acceptance

The harness must run both frameworks with actual durable storage and real child-process
kills, recover a correct receipt for each positive case, detect the duplicate-write control,
reject reused output directories, preserve failed trials and terminate only owned children.
Tests must exercise the service's key/payload conflict boundary and process protocol,
including a missing-barrier case that remains inconclusive. The full benchmark runs outside
the default fast suite and records its exact invocation and observed results.

Effects 0.3 will be a separately labelled adapter/profile once its public API is available.
Existing 0.2 results must not be represented as Effects results.
