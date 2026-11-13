# Validation record

Checked on 2026-11-13 on Windows, Python 3.13.18, against the local qagents 0.2.0
public API. No live model, Mem0 service, training controller, or optimizer was called.

| Check | Observed result |
| --- | --- |
| Source test suite | 68 passed |
| Installed-wheel test suite | 68 passed |
| Ruff lint | Passed for src, tests and examples |
| Ruff formatting | 17 Python files already formatted |
| qagents check_code | 20 Python source/test/example files and extracted documentation snippets checked; zero failures |
| README Python example | Executed successfully; reward 1.0 |
| CLI offline demo | Training reward 0.0; held-out rewards [1.0, 1.0]; outputs [700, 900]; frozen evaluation true |
| Wheel import and offline demo | Imported from an isolated installed-wheel target, with source path removed; passed |
| Distribution build | Wheel and source distribution built with hatchling 1.33.0 |
| Package contents | py.typed and dependency metadata present; local work/cache/database directories excluded |
| Staged source scan | No checked secret patterns or local user-specific paths found |

Build tooling was installed only in a Gym-local temporary environment. The
qagents environment was used for tests without installing or changing packages
there. Temporary files, caches and validation databases stayed inside ignored
`.work/`. No remote or publication was configured.

The automated suite covers SQLite persistence after reopen, namespace separation,
duplicate concurrent deltas, payload conflicts, transaction rollback, malformed
records/reflections, finite rewards, actual qagents tool traces and approval
pauses, held-out answer-key isolation, frozen evaluation, export round-trips,
and Mem0/Lightning fake-client contracts. Independent review found and prompted
fixes for an oversized numeric reward edge case, misleading concurrency wording,
an identifier-dependent test assertion, and unscoped Mem0 search results.

## Known limits

- The demo uses scripted behavior; its reward change does not establish real-model
  improvement or reproduce ACE benchmark results.
- This is ACE-inspired context learning. Official ACE package integration,
  semantic playbook deduplication and a social-consensus algorithm are not included.
- Peer/user corrections can be supplied through the evaluator interface. Their
  factual quality remains the evaluator's responsibility.
- SQLite memory uses lexical retrieval. Mem0 methods may call configured services;
  that integration has only fake-client validation.
- Identical delta replay is idempotent and batches are atomic. Distinct updates to
  one bullet use last-writer-wins; serialize training per namespace. The full
  model/curation/memory sequence does not provide transactional crash recovery.
- Evaluation logs episodes but does not adapt playbooks or memory. External
  callbacks and configured tools remain responsible for their own effects.
- Only plain-text qagents outputs are supported; the adapter does not resume
  approval pauses. qagents 0.3 compatibility remains untested until released.
- The Lightning bridge builds/sends reward events via injected transport.
  End-to-end RL integration and model-weight updates are not implemented.
  JSONL exports lack sampled token IDs/log probabilities required for policy RL.
- No remote service credentials were used. Code review and tests do not constitute
  an exhaustive security audit.
