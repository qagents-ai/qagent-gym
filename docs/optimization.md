# Evaluation and optimization exports

QAgent Gym now supplies two optional data boundaries without importing either
framework. They preserve the current qagents execution loop. No optimizer,
Inspect solver or live model has been run by these helpers.

## Inspect AI: evaluation datasets

`write_inspect_dataset` writes native JSONL sample fields: `input`, `target`,
`id` and `metadata`. Answer keys are supplied separately and are never appended
to task prompts. The dataset can be read by Inspect's documented
`inspect_ai.dataset.json_dataset`. See the
[official dataset contract](https://inspect.aisi.org.uk/datasets.html#dataset-samples).

```python
from qagent_gym.integrations import write_inspect_dataset
from qagent_gym.records import Task

write_inspect_dataset(
    [Task("test-7m", "Convert 7 metres to centimetres.", split="eval")],
    {"test-7m": "700"},
    "inspect-test.jsonl",
    partition="test",
    excluded_task_ids=["train-3m", "validation-5m"],
)
```

All targets must match task IDs exactly. Duplicate IDs, train tasks, declared
excluded IDs and existing output files are rejected before anything is written.
Use `partition="validation"` for optimizer selection and keep a separate final
test file. These metadata labels do not establish independence by themselves:
different IDs can still contain the same question or a near duplicate.

The answer key belongs in Inspect's dataset for scoring. A later solver must
continue passing only the task input to the generator. Native JSONL loading was
checked with `inspect-ai==0.3.289`, including Unicode, empty targets, sample order
and metadata, with `socket.socket.connect` disabled. Solver behavior and scorer
integration have not been tested.

The optional `tests/test_native_inspect.py` contract skips when Inspect is absent.
To run it in a separate environment with the selected Inspect version installed:

```powershell
python -m pytest tests/test_native_inspect.py tests/test_evaluation_exports.py -q
```

## GEPA: reflection data

`gepa_reflective_dataset(episodes, component="playbook")` returns a mapping from
the named text component to ordered reflection records. Each record contains
`Inputs`, `Generated Outputs`, `Feedback`, a score and provenance. These keys
follow the [GEPA adapter contract](https://gepa-ai.github.io/gepa/api/core/GEPAAdapter/).

Only completed training episodes are accepted. Duplicate episode IDs, mixed
evaluators and held-out episodes are rejected. Use an evaluator whose larger
scores mean better performance and whose scoring behavior stays fixed.

```python
import json

from qagent_gym.integrations import gepa_reflective_dataset

# training_episodes is a sequence returned by Gym.train, not Gym.evaluate.
payload = gepa_reflective_dataset(training_episodes)
with open("reflection.json", "x", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
```

GEPA defines a Python adapter return shape, not a native loader for this JSON.
A complete adapter must evaluate actual candidate text and construct aligned
[EvaluationBatch](https://gepa-ai.github.io/gepa/api/core/EvaluationBatch/)
outputs, scores and traces. This helper prepares reflection data; it does not
register an adapter, optimize a candidate or update model weights.

## Reproduce the local checks

```powershell
python -m examples.evaluation_exports --db .gym/evaluation-exports/gym.sqlite3
python -m pytest tests/test_evaluation_exports.py tests/test_examples.py -q
```

Keep three separate datasets for an optimization experiment: training feedback,
validation used for candidate selection, and an untouched final test set. Freeze
memory/playbooks across candidate comparisons. Compare against the same baseline
under the same call and token budgets. See [the ecosystem survey](ecosystem.md)
for the next implementation priorities and sources.
