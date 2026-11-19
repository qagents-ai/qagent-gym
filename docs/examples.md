# Offline recipes for the Quentin team

For larger application scenarios, see the [KA220 proposal workflow](erasmus-example.md),
[banking application sandbox](banking-example.md), and [English/Greek voice pipeline](voice-example.md).
The voice example additionally exercises optional local ASR and native TTS.

These examples use real qagents Agent loops with deterministic Python policies
through FunctionModel. They calculate answers, execute tools, review outputs and
reuse persisted lessons. They do not call hosted models, measure LLM quality,
or update model weights.

Run them from the repository root after installing qagent-gym and qagents:

| Recipe | Command | What it teaches |
| --- | --- | --- |
| Peer feedback | `python -m examples.peer_feedback` | Convert a separate reviewer's feedback into a reusable writing rule. |
| Tool regression | `python -m examples.tool_regression` | Inspect and correct tool arguments, then check another task and a regression case. |
| Frozen before/after | `python -m examples.before_after` | Evaluate exactly the same held-out tasks before and after training. |
| Integration contracts | `python -m examples.integration_contracts` | Supply a Mem0 client and Lightning transport explicitly, using local fakes first. |
| Evaluation exports | `python -m examples.evaluation_exports` | Separate Inspect test data from GEPA-shaped training reflection records. |
| Environment learning | `python -m examples.environment_learning` | Learn a multi-step review procedure, then evaluate with frozen memory. |
| Native environment replay | `python -m examples.environment_benchmark` | Replay seeded Gymnasium tasks through QAgents and LangGraph; install both extras. |

The learning recipes create temporary SQLite databases and remove them when finished.
To inspect the saved episodes, add `--db .gym/peer.sqlite3 --namespace writing-v1`,
using a different path/namespace for each experiment. The comparison recipe
requires an empty namespace; it refuses to silently reuse previous learning.
The fake integration recipe needs no database or command-line arguments.
The export recipe also removes its files by default; give it `--db` to retain them.

## A writer and a reviewer

`peer_feedback.py` gives Quentin the writer a feature, a benefit and a word limit.
The initial scripted policy adds promotional filler. Quentin the reviewer is a
separate qagents agent that actually checks the candidate's length, wording and
required content. `PeerEvaluator` converts that agent's JSON response into Reward.
The reflector produces a reusable style lesson; the next task changes the feature
and benefit while using the same lesson.

On a fresh namespace, the observed training rewards are **0, 1** and the held-out
reward is **1**. Example corrected output:

> Tool journaling helps teams audit tool calls.

The held-out task writes about durable sessions rather than repeating that answer.
The report includes the original text, actual reviewer feedback, corrected text,
held-out text and a check that evaluation left learning state unchanged.

To reuse the pattern, keep the typed Evaluator interface and replace the reviewer
with your own checker, user feedback source or configured reviewing agent. Replace
the scripted writing/reflecting models with configured models separately. Give a
model-based reviewer clear output instructions and validate its response. Peer
feedback is an extension point here; agreement or correctness is not guaranteed,
and no voting or social-consensus algorithm is implemented.

## Correct the call, then check for regressions

`tool_regression.py` separates the agent policy from a correct arithmetic tool.
The tool multiplies integer cents by quantity. The initial policy mistakenly
passes quantity 1 for every line, producing **325** instead of **775** cents.
After feedback, the playbook tells it to preserve requested quantities.

The held-out cart executes real `line_total` calls with `(150, 4)` and `(75, 2)`,
returning **600** and **150**, then totals **750** cents. A second cart with
single-item quantities still returns **300** cents. The report exposes tool-call
and tool-result steps, so the check covers changed arguments as well as final text.

For your own tools, replace the typed `line_total` function and task-specific
evaluator/reflector. Keep fixtures that exercise the failure and ordinary cases.
Mark a tool read-only only when it has no external effects. This recipe's policy
and reflector are explicit deterministic code, not learned model weights.

## Measure on the same held-out tasks

`before_after.py` defines two training tasks and four separate evaluation tasks.
It evaluates the four tasks with an empty playbook, trains only on the two
training tasks, and evaluates the original four again with frozen learned context.
Answer keys stay inside the evaluator. Evaluation episodes are logged but never
become training memory.

The measured result for this scripted conversion policy is:

| Held-out task | Expected | Before | After |
| --- | ---: | ---: | ---: |
| 7 minutes to seconds | 420 | 7 | 420 |
| 4 hours to minutes | 240 | 4 | 240 |
| 1 minute to seconds | 60 | 1 | 60 |
| 0 hours to minutes | 0 | 0 | 0 |
| Exact-match mean | | **0.25** | **1.0** |

The zero case already passes before training. Reporting that partial baseline is
intentional. The output records both task-ID lists, per-task results, state-freeze
checks and memory provenance. Running on two fresh databases returns identical
reports despite different internal episode UUIDs.

To study a real policy, preserve the same split and evaluation protocol, replace
the scripted models, and report the selected models and relevant sampling settings.
This small deterministic result is an integration check, not evidence of general
improvement or a reproduction of the ACE paper.

## Practice the integration boundary first

`integration_contracts.py` supplies its own FakeMem0 and fake transport. It stores
two memories in different namespaces, retrieves one scoped record with provenance,
and builds a Lightning reward event. The transport records the payload without
contacting its URL. The report contains no bearer credential.

This illustrates which objects to inject; it is not a Mem0 server emulator or a
Lightning training job. Replacing these fakes with real clients changes the
network behavior and requires explicit service configuration. See
[memory integration](memory.md) and [RL integration](rl.md) for the exact boundaries.

## Keep optimizer feedback separate from test truth

`evaluation_exports.py` runs the fixed-split example, then uses its actual logged
episodes to create two files beside the selected database:

- `.inspect.jsonl`: four unique held-out task inputs with independently supplied
  answer keys and `partition: test`. The duplicated pre/post episodes do not
  duplicate evaluation samples.
- `.gepa.json`: two completed training episodes in the `playbook` component's
  reflective data shape. Neither held-out outputs nor held-out feedback enters it.

```powershell
python -m examples.evaluation_exports --db .gym/export-recipe.sqlite3
```

The helper rejects an existing output path. Training task IDs are explicitly
excluded from the Inspect export. This generates data boundaries, not a finished
GEPA adapter or an Inspect evaluation run; a real optimization experiment still
needs candidate execution and separate validation/test handling. See
[optimization integration](optimization.md).

## Check the examples

```powershell
python -m pytest tests/test_examples.py -q
python -m ruff check examples tests/test_examples.py
python -m ruff format --check examples tests/test_examples.py
```

The tests run each module as a program, verify reuse on different tasks, inspect
actual tool arguments/results, check deterministic scores, and ensure held-out
episodes never enter learning memory. Recorded results are in
[validation-examples.md](validation-examples.md).
