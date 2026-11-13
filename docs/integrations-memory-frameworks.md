# Mem0, LangGraph and DSPy in QAgent Gym

These adapters accept explicitly configured clients or programs. Importing an adapter does not import its optional SDK, select a model, read a credential or start a service. Gym supplies its `Task` and immutable `Context`; successful adapters return a validated `Generation`. Provider exceptions propagate without automatic retries.

Checked on 13 November 2026 with Python 3.13: Mem0 SDK `mem0ai==2.2.1`, LangGraph `1.2.21`, `langchain-core==1.8.0` and DSPy `3.5.0`. The proposed compatible ranges are `mem0ai>=2.2.1,<3`, `langgraph>=1.2.21,<2`, `langchain-core>=1.8,<2` and `dspy>=3.5,<4`; native verification covers the exact versions above, not every later release in those ranges.

## Three executable examples

Run from a repository checkout with qagents and QAgent Gym installed. Only the latter two examples require their optional framework packages.

```console
python -m examples.mem0_memory
python -m examples.langgraph_agent
python -m examples.dspy_agent
```

| Example | Actual execution | Observed result |
| --- | --- | --- |
| `mem0_memory` | Real qagents `Agent`/`FunctionModel`, injected in-process Mem0 contract fixture | Learns a Greek release-note preference, captures it, then answers in Greek despite a later external change; zero remote reads/writes during evaluation |
| `langgraph_agent` | Native `StateGraph`, `InMemorySaver`, `interrupt()` and Gym | Cart calculation learns quantity handling: initial answer 175 fails; a different heldout cart returns 470 correctly; another task pauses with no exposed answer |
| `dspy_agent` | Native `dspy.Module.acall()` and `dspy.Prediction`, Gym | Learns an editorial rule on an export note and applies it to a different autosave note, removing promotional filler |

All policies are deterministic and run without model inference or a network service. The LangGraph and DSPy examples exercise those actual frameworks. The Mem0 example's fixture implements the SDK response contract and namespace filtering; it does not implement semantic retrieval. Native Mem0 plumbing is tested separately below. No optimizer, reinforcement-learning algorithm or weight update runs in these recipes.

## Mem0: scoped memory and explicit snapshots

`Mem0Memory(client)` implements Gym's synchronous `MemoryStore` interface. `add(record)` passes the exact text with `infer=False`, `user_id=record.namespace`, and metadata carrying the Gym memory ID and source episode. `search(query, namespace=..., limit=...)` uses `filters={"user_id": namespace}` and `top_k=limit`.

The adapter validates the returned committed results, text, namespace and provenance. Missing or mismatched user IDs, pending acknowledgements, malformed records and duplicate Gym memory IDs fail explicitly. Leading/trailing namespace whitespace is rejected before a call because the SDK trims entity IDs. A provided add-result namespace or provenance must match the requested values. A remote write may already have happened before an invalid or missing acknowledgement arrives; do not retry it blindly. Gym metadata is provenance, not a remote idempotency key.

The native Mem0 [Python SDK quickstart](https://docs.mem0.ai/open-source/python-quickstart) describes the `Memory` client and its result envelope. The [SDK implementation](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py) distinguishes `add(..., infer=False)` from inferred extraction and uses `search(..., filters=..., top_k=...)`. Hosted [search documentation](https://docs.mem0.ai/api-reference/memory/search-memories) describes a service API that can evolve separately. This adapter does not poll hosted jobs, accept pending writes as success or promise compatibility with all hosted result shapes.

Use an already configured synchronous client:

```python
from qagent_gym.integrations.mem0 import Mem0Memory

def attach_memory(configured_mem0_client):
    return Mem0Memory(configured_mem0_client)
```

Configure the real client's embedding provider, vector store, history path, telemetry and credentials explicitly. `infer=False` skips inference for extraction; it does **not** eliminate embedding computation or guarantee that an injected embedding provider makes no paid calls. The adapter neither creates a default `Memory()` nor changes your configuration.

Capture query results before evaluation:

```python
from qagent_gym.integrations.mem0 import Mem0Memory

def freeze_evaluation_memory(configured_client, evaluation_tasks, namespace):
    live = Mem0Memory(configured_client)
    return live.snapshot([task.prompt for task in evaluation_tasks], namespace=namespace)
```

Pass the returned `FrozenMem0Memory` as `memory=` to the evaluation Gym. Its `search` reads immutable, captured records only. Other namespaces, uncaptured queries, limits beyond the captured limit and all `add` calls fail. The capture performs sequential remote reads; it is not an atomic snapshot of a remote database. Once captured, changes to that database cannot change these evaluation inputs. Use the live adapter for training: the frozen adapter is intentionally read-only, and Gym training is not a transaction spanning local adaptation and remote memory writes.

## LangGraph: fresh runs with honest pause handling

`LangGraphGenerator(graph, output_key="output", recursion_limit=25, max_context_chars=20000)` accepts a compiled graph exposing `ainvoke`, `checkpointer` and, when checkpointed, `aget_state`. Each invocation supplies this state:

| Field | Value |
| --- | --- |
| `question` | `Task.prompt` |
| `context` | Rendered playbook and memories |
| `namespace` | Explicit Gym namespace |
| `task_id`, `split` | Task identity and train/eval split |

The adapter calls `ainvoke(..., version="v2", stream_mode="values")`. It expects `GraphOutput.value` to be a mapping with a nonempty text answer at `output_key`; it does not guess how to extract arbitrary agent message histories. Wrap a graph with another schema to map these fields into its inputs and back into an explicit output field. The current [v2 invocation contract](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/streaming.mdx) uses `value` and `interrupts`; a legacy dictionary result is rejected rather than silently misread.

Every call receives a fresh, short `thread_id`, with a namespace hash and random run identifier. Reusing a task ID does not resume a previous checkpoint. Thread IDs appear in the episode's first trace step for caller inspection. With a checkpointer, the adapter inspects the final checkpoint's pending nodes as well as dynamic interrupts. A static breakpoint after a provisional answer therefore remains `paused`, with an empty answer. Gym records that episode but does not reflect or learn from it.

LangGraph's [persistence guide](https://docs.langchain.com/oss/python/langgraph/persistence) separates checkpoint threads from long-term stores. This adapter isolates checkpoint runs; it does not enforce access control on a graph's own external store or prevent user-supplied nodes from changing state. Gym's evaluation freeze covers its captured input context. Use read-only application policies for graph-specific evaluation state. `InMemorySaver` in the example is process-local and is not restart durability. Paused checkpoints are not resumed here; an application that implements approval/resumption must coordinate that separately with LangGraph's [interrupt contract](https://reference.langchain.com/python/langgraph/types/interrupt). The recursion limit bounds graph steps, not wall-clock time, cost or node side effects.

## DSPy: invoke a supplied program and validate its prediction

`DSPyGenerator(program, output_field="answer", max_context_chars=20000)` calls `await program.acall(question=task.prompt, context=...)`. The context argument is JSON containing `namespace`, `split` and the rendered `learned_context`. The program must support `acall`, normally through a DSPy module implementing `aforward`, and return a `Prediction` with a nonempty string at the configured output field.

Only that validated answer enters the journal; arbitrary extra prediction objects are not serialized. Missing fields, invalid types and module failures raise rather than manufacture successful generations. DSPy does not provide a general graph-pause result through this interface; use a separate workflow layer when approval/resumption is required. Consult the native [Module API](https://dspy.ai/current/api/modules/Module/) and [Prediction API](https://dspy.ai/current/api/primitives/Prediction/).

A configured `dspy.Predict` or custom program with matching inputs can replace the deterministic module. Its language model and provider credentials remain the caller's responsibility. The adapter does not configure global DSPy settings, compile a program, mutate model weights or invoke an optimizer. Program-specific mutable state and side effects remain outside Gym's context freeze.

## Verification and limits

The fast tests use injected contract fixtures and skip SDK-specific tests when those SDKs are absent. The separate native tests instantiate the exact versions above without using a model service:

- Mem0: actual `Memory.from_config`, `Memory.add`, `Memory.search`, SQLite history and local in-memory Qdrant; only the embedding and LLM factories are replaced with deterministic test providers. The LLM substitute raises if called. This checks SDK plumbing and provenance, not embedding quality, cloud APIs or inference.
- LangGraph: an actual compiled graph, a dynamic interrupt and a static breakpoint after a provisional answer; tests also verify fresh thread identities, malformed output rejection, provider errors and no learning from paused training.
- DSPy: an actual `Module`, `acall`, `aforward` and `Prediction`; no LM, optimizer or hosted service.

```console
python -m pytest tests/test_mem0_adapter.py tests/test_langgraph_adapter.py tests/test_dspy_adapter.py tests/test_integrations.py -q -p no:asyncio
```

Native checks were run in an isolated repository-local environment. Cache/history/temp paths were confined to that environment, tracing and telemetry were disabled, and credential-like environment variables were removed. Local Qdrant emits its expected warning that payload indexes have no effect in local mode; no server index performance was tested. Mem0 is Apache-2.0; LangGraph, langchain-core and DSPy are MIT according to the installed distribution metadata. No upstream implementation code is vendored by these adapters.

Measured validation: 85 fast tests passed with four optional native tests skipped (including the existing integration-contract suite); all 48 tests in the three new adapter test files passed with the native SDKs installed. Ruff checks and all three example CLIs passed. The native run emitted only the local-Qdrant warning described above. A separate read-only review found no remaining blockers.
