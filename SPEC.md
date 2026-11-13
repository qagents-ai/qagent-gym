# QAgent Gym 0.1

## Objective and scope

A separate Python 3.11+ library that uses the public qagents API to run tasks,
evaluate results, and retain useful experience between runs. The first release
implements ACE-inspired context adaptation, not the official ACE package or
model-weight training. Default demos and tests require no network or credentials.

## Capability map and build order

| Module | Contract | Depends on |
| --- | --- | --- |
| records | Frozen Task, Context, Generation, Reward, Episode and PlaybookDelta values | stdlib |
| storage | SQLite scoped memory, playbook snapshots and atomic delta application | records |
| gym | Generator -> evaluator -> reflector -> curator loop | records, storage |
| qagents | Public Agent.run adapter and bounded scripted example | gym, qagents |
| integrations | Injected Mem0 adapter and pure Lightning reward bridge | records |

Build records/storage first, then the runnable learning loop, then integration
contracts and documentation. One worker owns core; a delegated worker may own
integrations and their tests. No changes to qagents itself.

## Core contracts

- Task contains task_id, prompt and split (train/eval). Evaluator-only targets
  stay in the evaluator; the generator receives only the prompt and context.
- Context contains immutable playbook bullets and retrieved memory. Generator
  exposes async generate(task, context) -> Generation; no mutable store is exposed.
- Generation records status, final output, observable steps and model identity.
  An incomplete result cannot be represented as a successful answer.
- Reward contains a finite numeric value, evaluator identifier and feedback.
- Episode freezes task, generation, reward, input context and run_id. JSONL
  schema version 1 round-trips without arbitrary object deserialization.
- Reflector exposes async reflect(episode) -> tuple[Lesson, ...]. Curator
  converts lessons into PlaybookDelta values. Both can be user-supplied.
- A delta has a stable delta_id, bullet_id, add/update/deprecate operation,
  text, source episode ID and helpful/harmful counts. SQLite applies each delta
  transactionally. Identical replay is a no-op; changed payload with reused ID
  fails. Unknown update/deprecate and conflicting add fail without partial writes.
- Memory is explicit text with namespace and provenance. SQLite uses local
  lexical retrieval; Mem0 accepts an already configured client and normalizes
  its current add/search response shapes. No SDK is imported automatically.
- Evaluation takes a frozen context snapshot, never invokes reflection/curation
  or writes memory/playbook. Training and evaluation namespace are explicit.
- Gym records a completed episode before adaptation. Training persists a compact
  output/feedback memory; generated lessons remain auditable in the delta log.
- Exceptions propagate; a crashed run is not represented as successful. This
  release does not promise transactional recovery of an entire model episode.

## Commands and layout

Source: src/qagent_gym; tests: tests; runnable demo: python -m qagent_gym demo.

```powershell
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m qagent_gym demo --db .gym/demo.sqlite3 --export .gym/episodes.jsonl
```

Use typed Python, frozen dataclasses for records, explicit keyword arguments
and ValueError for invalid boundary values. Example: Reward(value=1.0,
evaluator="exact-v1", feedback="Matched the expected answer.").

## Acceptance

1. The real qagents Agent loop is exercised offline using FunctionModel or
   ScriptedModel, finite Limits, and checks result.status before output.
2. A scripted failed training task produces a persistent lesson; a later run
   reads the learned context and succeeds. This checks the integration, not LLM quality.
3. SQLite state survives close/reopen, separates namespaces, rejects conflicting
   delta IDs, and supports atomic/idempotent batches including concurrent callers.
   Distinct updates use last-writer-wins; training jobs must serialize per namespace.
4. A held-out run leaves memory, playbook revisions and delta log unchanged.
5. Episode values are immutable; malformed JSONL, non-finite rewards and invalid
   deltas are rejected. Successful exports round-trip.
6. Optional Mem0 and Lightning boundaries are tested with fake clients/transports.
   Real services and RL optimizers are explicitly untested.
7. CLI works from a fresh database and prints a small machine-readable report.
8. Tests, lint, formatting, package build and qagents docs checks pass.

## Boundaries

Implement only this authorized repository; no paid calls, model downloads, remote
publication, secrets, global configuration, or qagents environment modifications.
Keep optional frameworks outside base dependencies. Initialize local git without
committing until review. A future real RL run requires a selected trainable model,
dataset, reward and Linux/WSL training stack; exports alone are not RL training.
