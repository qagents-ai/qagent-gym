# QAgent Gym

Persistent context learning and frozen evaluation for **qagents**. Run an agent,
evaluate its result, reflect on feedback, and apply incremental playbook changes.
The next task receives the learned playbook and relevant memories.

This first release implements an **ACE-inspired** generator/reflector/curator loop.
It does not depend on either package named `ace`, reproduce the ACE paper's
benchmarks, or update model weights. The offline demo uses qagents' actual Agent
loop with scripted models; it checks the integration without credentials or network.

## Run the offline demo

Requires Python 3.11+ and qagents 0.2.x or 0.3.x. The validated checkout is 0.2.0;
compatibility with an eventual 0.3 release must be checked before upgrading.
Install your local qagents checkout first if it is not published on your index:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e 'C:\path\to\qagents'
python -m pip install -e '.[dev]'
python -m qagent_gym demo --db .gym/demo.sqlite3 --export .gym/episodes.jsonl
```

A fresh namespace reports `training_reward: 0.0`, `evaluation_rewards: [1.0, 1.0]`
and `frozen_evaluation: true`. The training task teaches a metres-to-centimetres
rule. The database closes and reopens before two different tasks use that rule.
Reusing a database intentionally retains learning; use `--namespace another-run`
for a fresh experiment. Export refuses to replace an existing file.

```powershell
python -m qagent_gym export --db .gym/demo.sqlite3 --namespace metres-demo --output .gym/history.jsonl
python examples/offline_learning.py
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m hatchling build
```

## Use your own qagents agents

`QAgentsGenerator` takes an existing text-output `Agent`. It calls public
`Agent.run`, bounds turns/time/tokens, starts a fresh session per task, checks
`result.status`, and captures observable messages and tool calls/results.
The caller owns model connections and should close them when done. Supplied
agents may contact their configured providers; only the scripted demo is offline.

```python
import asyncio

from qagents import Agent
from qagents.models.testing import ScriptedModel

from qagent_gym.gym import ExactMatchEvaluator, Gym
from qagent_gym.qagents_adapter import QAgentsGenerator, QAgentsReflector
from qagent_gym.records import Task
from qagent_gym.storage import SQLiteStore


async def main():
    solver = QAgentsGenerator(Agent(ScriptedModel(["wrong", "right"])))
    reviewer = QAgentsReflector(QAgentsGenerator(Agent(ScriptedModel([
        '[{"bullet_id":"verify","text":"Check the result before answering."}]'
    ]))))
    evaluator = ExactMatchEvaluator({"practice": "right", "heldout": "right"})
    with SQLiteStore("learning.sqlite3") as store:
        gym = Gym(solver, evaluator, reviewer, store)
        await gym.train(Task("practice", "Solve the practice task"), namespace="example")
        results = await gym.evaluate([
            Task("heldout", "Solve a different task", split="eval")
        ], namespace="example")
        print(results[0].reward.value)


asyncio.run(main())
```

The minimal `Generator`, `Evaluator`, `Reflector`, `Curator`, and `MemoryStore`
protocols are in `gym.py`. An evaluator can return objective feedback, a user's
correction, or a peer agent's review as a typed `Reward`. That feedback reaches
the reflector only after the attempt. A peer-review callback is a supported
extension point; no consensus or social-learning algorithm is implemented.

## Persistence and evaluation contract

- SQLite stores scoped memory, playbooks, change history and immutable episodes.
  Memory retrieval uses case-insensitive word overlap, not embeddings.
- Playbook deltas use stable IDs, provenance and helpful/harmful counters.
  Identical replay is a no-op; reusing an ID with another payload fails. Batches
  commit atomically. Distinct updates use last-writer-wins, so serialize training
  jobs sharing a namespace. Semantic deduplication is not implemented.
- `evaluate` freezes all input contexts before starting. It never reflects,
  curates or adds memory. Evaluation episodes are still logged. Answer keys live
  in the evaluator and are not sent to the solving agent.
- Learning uses the supplied feedback; it does not establish that a lesson is
  true. Keep held-out tasks separate and compare the same tasks before/after
  adaptation when measuring quality. Learned text is explicit model input.
- Exceptions propagate. An episode is saved before adaptation, but the entire
  model/memory/curation sequence is not a crash-resumable transaction. A failure
  may leave an episode without learning, or a playbook update without memory.
- Paused/blocked/limited runs expose no final answer and do not adapt. The adapter
  does not resume approvals. JSONL contains observable traces and feedback;
  it lacks the sampled token IDs and log probabilities needed for policy RL.
- The adapter rejects context larger than 20,000 characters by default instead
  of silently truncating it. Scope memories or deprecate irrelevant bullets.

## Optional integrations

[Mem0](docs/memory.md) is an injected, validated adapter for a preconfigured client.
It is outside base dependencies. [Agent Lightning and RL](docs/rl.md) documents
the v1 gateway route, Linux/WSL constraints, TRL/verl alternatives, and a pure
reward-event bridge. Fake-client tests validate these boundaries; no live Mem0
service, Lightning controller or optimizer has been run.

See [SPEC.md](SPEC.md) for contracts and acceptance criteria and
[sources and licenses](docs/sources.md) for upstream references.
