# Example validation

Checked on 2026-11-13 on Windows with Python 3.13.18 and qagents' public 0.2.0
Agent API. The qagents owner provided a stable source window for these checks.

| Check | Observed result |
| --- | --- |
| `python -m pytest tests/test_examples.py -q -p no:asyncio` | 10 passed |
| `python -m ruff check examples tests/test_examples.py` | Passed |
| `python -m ruff format --check examples tests/test_examples.py` | 9 Python files formatted |
| qagents `check_code` | All 9 example/test Python files passed |
| Module entry points | All five new recipes executed as subprocesses, returned valid JSON and wrote no stderr |

Each subprocess used a temporary directory. No hosted model, Mem0 service,
Lightning controller, Inspect evaluator or GEPA optimizer was invoked. No model
weights or additional dependencies were downloaded for these examples.

## Recorded behavior

- **Peer feedback:** training rewards `[0, 1]`, held-out reward `1`. A separate
  qagents reviewer identified excessive wording. The next, different writing task
  used the persisted style rule; evaluation left learning state unchanged.
- **Tool regression:** initial cart answer `325` cents versus expected `775`.
  After adaptation, actual tool arguments preserved quantities 4 and 2; results
  were `600` and `150`, summed to `750`. A single-item regression cart returned
  `300`. Both held-out rewards were `1`.
- **Frozen comparison:** the same four held-out tasks scored `0.25` before
  training and `1.0` afterward. Both evaluation passes preserved playbook/memory
  state. Two fresh databases produced equal reports. All stored/retrieved learning
  memories referenced training episodes, not held-out episodes.
- **Fake integrations:** retrieval returned one memory from the requested namespace
  with its original provenance. Exactly one reward event reached the injected
  fake transport; no credential appeared in the report.
- **Evaluation exports:** four distinct Inspect test rows contained independently
  supplied targets. Two completed training episodes entered GEPA-shaped reflection
  data. Held-out episode IDs were absent. Reusing output paths was rejected.

The scores describe intentionally limited deterministic policies. They check
dataflow, correction, task separation, tool calls and export contracts; they do
not measure pretrained-model quality, establish generalization beyond these
fixtures, or demonstrate model-weight training. The export helpers do not supply
a complete Inspect task/solver or GEPA optimization adapter.
