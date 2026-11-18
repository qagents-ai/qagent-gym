# Start here

QAgent Gym runs practice tasks and benchmarks against an agent you can already
run. QAgents executes the model and tools. Gym supplies context, scores the
answers, and keeps a record of what the agent learned.

## Learn from feedback

Run the [offline demo](../README.md#run-the-offline-demo) from a fresh database.
It first asks a scripted agent to convert metres to centimetres. The agent
gets the practice question wrong. The evaluator gives feedback, and Gym saves
a lesson in the playbook. After Gym reopens the database, the agent uses the
lesson to solve two new tasks.

| Term | What it means in the demo |
| --- | --- |
| Task | A conversion problem sent to the agent |
| Reward | A numeric score with feedback from the evaluator |
| Episode | The task, attempt, score and context stored together |
| Playbook | Reusable instructions, such as the conversion rule |
| Memory | Facts or observations retrieved for a task |

See [how the learning loop stores context](diagrams/README.md#learning-and-evaluation)
and [how to scope memory to a project](memory.md).

## Measure a change

Take a baseline on held-out tasks. Run the same tasks with your candidate
configuration and keep the scorer, inputs, and limits fixed. Compare results
with the same playbook and memory if you are changing the model or prompts.
When measuring a playbook change, freeze each version for its entire batch.

```console
python -m examples.before_after --db .gym/start-here/comparison.sqlite3
```

This scripted example runs four held-out tasks before and after learning. Its
mean score moves from `0.25` to `1.0`. Both evaluation batches leave the
playbook and memory unchanged. See [the recipe](examples.md) for the individual
tasks and how to choose a fresh namespace.

## Bring your own agent

Follow [Run real agents](real-agents.md) to select a model endpoint, test an
approval or delegate a task. Wrap that agent with `QAgentsGenerator` to use it
in Gym. Choose an evaluator from signals you can verify, such as an exact
answer, a test result or a checked tool response.

For a complete example, try the [Erasmus+ KA220 proposal](erasmus-example.md),
the [banking sandbox](banking-example.md), or the [bilingual voice agent](voice-example.md).

## Export results or explore model training

Use [evaluation exports](optimization.md) to produce Inspect AI datasets or
GEPA reflection data. Start with a baseline and keep training, validation and
final test tasks separate. If your goal is to update model weights, the
[reinforcement learning guide](rl.md) shows which additional rollout data a
trainer needs and how to preview a reward from a local episode.
