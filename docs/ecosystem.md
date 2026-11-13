# QAgent Gym ecosystem survey — 2026-11-13

Recommendation: prioritize independent evaluation and measured context optimization,
then optional semantic memory and a separately deployed RL trainer. Keep qagents'
journal, permissions and retry semantics authoritative. Adding dependencies alone
does not establish state-of-the-art performance.

## Search boundaries and evidence

This was a targeted technical survey of official repositories, API documentation,
release metadata and two original research papers, checked on 2026-11-13. Search
terms combined library names with adapter, evaluator, rollout, memory, OpenTelemetry,
license and version. Current English documentation was screened; historical material
was used only to identify incompatible older APIs. GitHub metadata was queried for
12 candidates; observability licenses and the alternate ACE implementation were
checked separately. No third-party rankings, star counts or marketing comparisons
were treated as evidence. No library installation, live service experiment or
independent benchmark replication was performed by this research task.

Evidence grades: **A** = implementation, official schema or release/license file,
strong evidence of the inspected contract; **B** = original author paper, evidence
for its reported experiment only. The [ACE paper](https://arxiv.org/abs/2510.04618v3)
and [GEPA paper](https://arxiv.org/abs/2507.19457) are B for this survey. Their benchmark
results do not establish gains on Quentin's tasks. These are original claims, not
independent confirmations; this survey did not assess the replication literature.
API fit is high-confidence where source was inspected; benefit to this Gym remains
unmeasured. Rolling docs can be newer than the tagged release.

## Ranked options

| Priority | Library / checked state | Concrete fit | Boundary and decision |
| --- | --- | --- | --- |
| 1 | **Inspect AI**, GitHub release **0.3.289** (Nov 10), MIT; A | Independent task datasets, scorers and evaluation logs. Native JSONL samples use `input`, `target`, `id`, `metadata`; a later custom solver can await the existing Gym generator and return Inspect state. | Add the dependency-free dataset exporter now. Keep answer keys separate from generator inputs. A real solver requires installed-version integration tests and concurrency/budget checks. [Dataset contract](https://inspect.aisi.org.uk/datasets.html), [solver contract](https://inspect.aisi.org.uk/solvers.html), [release](https://github.com/UKGovernmentBEIS/inspect_ai/releases/tag/0.3.289), [license](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/LICENSE) |
| 2 | **GEPA**, **0.1.4**, MIT; A | Optimize named prompt/playbook components while retaining qagents execution. Its custom adapter evaluates batches and constructs reflective feedback; the newer `optimize_anything` API accepts an evaluator for a text artifact. | Add a train-only reflective-data export now; later build a budgeted candidate evaluator using fresh/frozen contexts. GEPA uses validation data for selection, so reserve a third untouched test split. Reflection/model evaluation can incur provider costs. [Adapter](https://gepa-ai.github.io/gepa/api/core/GEPAAdapter/), [batch](https://gepa-ai.github.io/gepa/api/core/EvaluationBatch/), [optimizer](https://gepa-ai.github.io/gepa/api/core/optimize/), [release](https://github.com/gepa-ai/gepa/releases/tag/v0.1.4) |
| 3 | **OpenTelemetry**, Apache-2.0; optional **Phoenix** server **21.1.0**, ELv2; A | Reuse qagents' existing OTel hooks. Correlate Gym episode, context revision, reward, duration and qagents spans, exporting to a user-selected OTLP collector. | Prefer vendor-neutral spans over a hard dependency on a hosted dashboard. GenAI agent conventions moved to a separate repo and remain Development. Phoenix accepts OTLP and self-hosts, but its server license is ELv2 with hosted-service restrictions. Langfuse is another collector option: core MIT, enterprise directories separately licensed. [Current agent conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md), [Phoenix OTLP](https://arize.com/docs/phoenix/learn/tracing/how-tracing-works), [Phoenix license](https://github.com/Arize-ai/phoenix/blob/main/LICENSE), [Langfuse license](https://github.com/langfuse/langfuse/blob/main/LICENSE) |
| 4 | **Official ACE**, main metadata **0.1.0**, Apache-2.0, no GitHub release; A/B | Reference generator/reflector/curator behavior, incremental playbook edits, bullet usage and optional deduplication. Its `ACE.run` supports offline, online and eval-only modes. | Continue describing this Gym as ACE-inspired. Official ACE constructs provider clients; offline adaptation means dataset adaptation, not no network. A direct adapter would require explicit client injection and migration/provenance tests. Do not conflate it with Kayba `ace-framework` **0.13.0**, Python >=3.12; both expose an `ace` import. [Official code](https://github.com/ace-agent/ace), [metadata](https://github.com/ace-agent/ace/blob/main/pyproject.toml), [license](https://github.com/ace-agent/ace/blob/main/LICENSE.txt), [Kayba metadata](https://github.com/kayba-ai/agentic-context-engine/blob/main/pyproject.toml) |
| 5 | **Mem0**, Python metadata **2.2.1**, Apache-2.0; A | Existing injected MemoryStore adapter is the right seam for semantic memory. Current local SDK adds with `user_id`, searches with `filters={user_id: ...}` and `top_k`, and returns a results envelope. | Keep SQLite as default. Add native-package contract coverage before advertising live compatibility. `infer=False` avoids extraction, not necessarily embedding calls. Hosted asynchronous acknowledgements differ. GitHub's latest release is a TypeScript tag, not the Python version. [Pinned implementation](https://github.com/mem0ai/mem0/blob/f1791daa86d469249c9b002f62644d58a1eb96a0/mem0/memory/main.py), [metadata](https://github.com/mem0ai/mem0/blob/main/pyproject.toml) |
| 6 | **DSPy**, release **3.5.0** (Nov 10), MIT; A | Useful when optimizing several typed predictors or an existing DSPy program. `dspy.GEPA(...).compile(student, trainset, valset)` connects feedback and structured predictors. | Optional integration, not a replacement for the qagents runtime. For this small framework, direct GEPA is the smaller initial dependency boundary. Pin the metric signature; current source supports predictor and program trace feedback. [Release](https://github.com/stanfordnlp/dspy/releases/tag/3.5.0), [pinned GEPA source](https://github.com/stanfordnlp/dspy/blob/3.5.0/dspy/teleprompt/gepa/gepa.py) |
| 7 | **Agent Lightning**, release **1.0.3** (Nov 3), MIT; A | Best future route to policy-weight RL while retaining the qagents agent loop: redirect its OpenAI-compatible model endpoint through the gateway, let the controller manage rollout execution, and attach terminal rewards. | Existing reward bridge is preparation only. v1 differs from old LitAgent/Trainer/emit_reward examples. Native Windows local controller is unsupported; use Linux/WSL or Kubernetes. Pin a tested stack: even the current stable URL displays a development notice. [Architecture](https://microsoft.github.io/agent-lightning/stable/05-basics/), [platform](https://microsoft.github.io/agent-lightning/stable/30-controller-configuration/), [release](https://github.com/microsoft/agent-lightning/releases/tag/v1.0.3) |
| 8 | **TRL 1.17.1** and **verl 0.9.1**, Apache-2.0; A | Alternative policy-training backends once a trainable model, tokenizer, task distribution and GPU stack are selected. TRL offers custom GRPO rollout functions; verl exposes an agent loop. | TRL's rollout hook is experimental and needs sampled token IDs/log probabilities and correct generation multiplicity. verl's documented agent-loop API is alpha and needs prompt/response IDs plus a tool-output mask. Plain Gym JSONL cannot reconstruct these. Prefer Lightning first if preserving the agent loop matters. [TRL contract](https://huggingface.co/docs/trl/grpo_trainer), [verl contract](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), [TRL release](https://github.com/huggingface/trl/releases/tag/v1.17.1), [verl release](https://github.com/verl-project/verl/releases/tag/v0.9.1) |
| 9 | **LangMem**, main metadata **0.0.30**, MIT; A | Alternative structured memory extraction or prompt refinement when an application already uses LangChain/LangGraph. `create_memory_manager` can return typed memory changes while the caller controls storage. | A reasonable optional reflector/memory adapter; unnecessary base dependencies for this Gym. Extraction uses LLM calls. Keep namespace and deletion policies explicit. [Functional memory guide](https://langchain-ai.github.io/langmem/guides/extract_semantic_memories/), [metadata](https://github.com/langchain-ai/langmem/blob/main/pyproject.toml), [license](https://github.com/langchain-ai/langmem/blob/main/LICENSE) |
| 10 | **Gymnasium 1.4.0**, MIT; **BrowserGym 0.14.3**, Apache-2.0; A | Reusable environments for a future browser or simulated-tool curriculum. BrowserGym registers Gymnasium tasks and exposes reset/step with observations, reward, terminated and truncated. | Add only for an actual environment benchmark. Browser setup, browser processes and benchmark services are additional infrastructure; the name QAgent Gym does not imply Gymnasium compatibility. Preserve termination versus truncation and close environments. [Gymnasium API](https://gymnasium.farama.org/main/content/basic_usage/), [BrowserGym loop](https://github.com/ServiceNow/BrowserGym#usage), [BrowserGym license](https://github.com/ServiceNow/BrowserGym/blob/main/LICENSE), [Gymnasium release](https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.4.0) |

## Data boundaries implemented in this update

The two helpers below are implemented in `qagent_gym.integrations.evaluation`,
with local contract tests. They do not run either third-party framework.
See [usage and limits](optimization.md).

### Inspect dataset export

Emit one JSON object per task, using a separately supplied answer mapping:

```json
{"input":"Task prompt","target":"Expected answer","id":"task-1","metadata":{"qagent_gym_split":"eval"}}
```

Use Inspect's existing `json_dataset(path)` reader. The exporter validates IDs,
requires separately supplied targets, rejects duplicate and declared excluded task
IDs, preserves order, refuses replacement, and keeps target text out of `input`.
Schema tests run without Inspect. They do not
establish end-to-end solver or scorer compatibility.

### GEPA reflective-data export

The adapter API returns a mapping from component name to records. The documented
record keys are a recommended schema, not a universal file-loader protocol:

```json
{"playbook":[{"Inputs":{"task":"Task prompt"},"Generated Outputs":"Attempt","Feedback":"Evaluator feedback","score":0.0,"episode_id":"run-1"}]}
```

The exporter accepts completed train episodes with finite rewards, unique episode
IDs and a single evaluator; it keeps one aligned record per episode. Extra provenance
fields are allowed. A later
adapter must construct `EvaluationBatch` outputs/scores/optional trajectories in
the same order and use candidate text when running its generator. Exporting this
JSON alone neither runs GEPA nor registers a compatible optimizer adapter.

## Evaluation needed before a stronger quality claim

Use fixed disjoint train, optimizer-validation and untouched test task IDs; freeze
memory/playbooks for every candidate comparison. Persist dataset fingerprints,
model/version and sampling configuration, context revisions, seeds, call/token
budgets, latency and paired per-task rewards. Compare baseline, memory-only,
ACE-inspired and GEPA variants under the same budgets. Report uncertainty and
failure/approval/timeout rates; do not select on the final test split. This is a
recommended experiment design, not a completed experiment.

Inspected source heads (audit anchors): GEPA ce5b0d6e3d6af6c6dbec6de2c54536bbb961d309
(Nov 9), Inspect e4435baa530d05324358bedc3782609c952f9876 (Nov 12), ACE
82709de050e1db6e6ef2f07bcb0393560b94992a (Aug 24), Mem0
f1791daa86d469249c9b002f62644d58a1eb96a0 (Nov 6), LangMem
8620b986af5b0c7593e6b7fbece2998690d22d15 (Nov 3), Lightning
2070bad5113e52f1aa2e5bdba822d4b24b197c05 (Nov 12). Release metadata and source heads
were read from the corresponding official GitHub APIs on the survey date.
