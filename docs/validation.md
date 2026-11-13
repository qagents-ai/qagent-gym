# Validation record

## Application scenarios update

Checked on 2026-11-13 against qagents' frozen stage-0 runtime at `4380241`.

| Check | Observed result |
| --- | --- |
| Combined source suite | 168 passed, 1 optional Inspect test skipped, in 9.16 seconds |
| Installed-wheel suite | 168 passed, 1 optional Inspect test skipped, in 8.64 seconds; wheel import and offline demo passed |
| KA220 workflow | 38 tests; executed 3,201-word draft, 75 document checks, 7 initial errors to 0 final errors, 6 evidence warnings remain |
| Banking application | 13 tests; actual CLI/tool execution, rollback, ownership and concurrent-transfer checks |
| Voice pipeline | 15 tests; interruption, caller cancellation, partial writes, WAV bounds, worker timeouts and bilingual qagents replies |
| Native voice run | 6 actual ASR/TTS trials; median first response WAV 483 ms English / 555 ms Greek; zero WER on two synthetic input phrases |
| Ruff lint and formatting | Passed; 37 Python files formatted |
| qagents check_code | 42 Python files and documentation snippets; zero failures; changed Erasmus files also rechecked after review fixes |
| Packaging | Wheel and source distribution built; source distribution includes application fixtures and native speech worker |

The voice measurement includes audio validation through first-WAV creation and
reports startup separately. It excludes recording, playback, network and a live
LLM. Native ASR/TTS dependencies were installed in an isolated Gym-local environment;
the qagents environment stayed unchanged. Initial native execution exposed a
faster-whisper/PyAV API mismatch; the tested compatible versions are pinned in the
optional speech requirements. Review findings prompted red/green regressions for
partial-file cleanup and cancellation propagation, plus additional bounds and
deadline checks. See [voice measurements and reproduction](voice-example.md).

The Erasmus+ output is a fictional worked planning draft with explicit evidence
gaps. Its final evaluation revises the same brief and does not establish held-out
generalization. The banking sample uses fictional TEST units and a host-injected
principal; it is not a production financial service. All qagents policies in these
examples are deterministic. See [Erasmus verification](erasmus-example.md) and
[banking checks](banking-example.md).

Final review also exposed a proposal-output race. Before correction, concurrent
runs could share a database or overwrite artifacts created while planning was
awaited. Six new regressions failed on that implementation. The corrected recipe
claims its destination exclusively before agent work and creates its database
and artifacts exclusively; 38 Erasmus tests now pass, including contention and
preservation of externally created files.

## Quentin team examples and evaluation exports update

Checked on 2026-11-13 against qagents' public API. The final source and installed-
wheel suites ran against frozen stage-0 runtime code at commit `aeecaea`.

| Check | Observed result |
| --- | --- |
| Combined source suite | 102 passed, 1 optional Inspect test skipped, in 5.74 seconds |
| Installed-wheel suite | 102 passed, 1 optional Inspect test skipped, in 5.48 seconds |
| New export contracts | 24 passed |
| New examples | 10 passed; all five module entry points executed |
| Native Inspect reader | inspect-ai 0.3.289; native reader plus export suite: 25 passed in 3.33 seconds, `socket.socket.connect` disabled |
| Ruff lint and formatting | Passed; 28 Python files formatted |
| qagents check_code | 33 Python files and documentation snippets; zero failures |
| Package build | Wheel and source distribution built with isolated Gym-local tooling |
| Installed wheel | New export functions imported from the wheel target; original offline demo passed |
| Artwork | Two generated PNGs inspected for labels, composition and context-learning meaning |

Independent review found a Unicode preflight issue in the Inspect exporter.
Three tests demonstrated partial files for malformed second-row input, target
and ID values. The exporter now validates UTF-8 bytes before creating a file;
all three regressions pass. Two further malformed exclusion-list regressions
also failed before validation was tightened. The initial wheel test harness
could not find repository examples; adding the repository root (without src) to
the harness path fixed collection, and the complete wheel suite then passed.

Inspect was subsequently installed in an isolated Gym-local environment to test
its native dataset reader. The optional reader test skips in environments without
Inspect; the existing 102 tests do not require it. GEPA was not installed or
executed. The export contracts follow current official documentation; a live solver
or optimizer remains separate work. No real-model quality or RL result is claimed.
The source and wheel tests used the existing qagents environment without modifying
its packages. See [the example validation](validation-examples.md),
[optimization boundaries](optimization.md), and [art prompts](../assets/README.md).

## Initial implementation

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
