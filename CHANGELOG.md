# Changelog

## Unreleased

- Add five executable offline recipes covering peer feedback, tool regression,
  frozen before/after evaluation, injected Mem0/Lightning clients and data exports.
- Add Inspect JSONL export with separate answer keys, ID alignment, explicit
  validation/test labels, exclusion checks and serialization before file creation.
- Add training-only GEPA reflection data with aligned scores, feedback and provenance.
- Add Quentin team hero and learning-loop illustrations, with saved generation prompts.
- Document the current ecosystem, upstream API and license boundaries, and the
  experiments needed before claiming real-model improvements.
- Validate 102 tests from source and from the installed wheel.

## 0.1.0 — initial local implementation

- ACE-inspired generator/evaluator/reflector/curator loop using public qagents APIs.
- Scoped SQLite memory, versioned playbooks and immutable episode records.
- Frozen evaluation, JSONL episode exchange, optional Mem0 and Lightning boundaries.
- Offline CLI, package metadata, and 68 initial tests.
