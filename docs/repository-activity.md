# Evaluate an agent that answers from repository history

The [QAgents repository activity example](https://qagents-ai.github.io/qagents/guides/repository-activity/)
indexes Git commits and optional Actions outcomes before answering. It supports
questions such as "What changed yesterday?" and "Τι έγινε την προηγούμενη εβδομάδα;".
SQLite time indexes and full-text search serve the questions without another Git
walk or GitHub request.

Use the runnable commands in the QAgents guide to create the index, ask questions
and measure lookup latency. The no-inference path returns deterministic evidence.
Agent mode runs the same snapshot through a read-only QAgents tool, with an
offline routing fixture or an explicitly selected model provider.

## Evaluate answers against a fixed snapshot

Keep the indexed head, timezone, as-of timestamp and CI snapshot fixed when
comparing agents. The expected answer should record the exact time window,
matching commit identities, CI outcomes and any unknown or partial coverage.

Useful held-out cases include:

- A commit just before and just after local midnight.
- A daylight-saving transition and a previous calendar-week question.
- No matching commits, no imported CI data, or a failed CI run.
- A commit title containing instructions the model should treat as evidence.
- A refreshed or rewritten ref whose old results must disappear.

The example already verifies counts and citations in its QAgents output validator.
A Gym evaluator should separately assess the prose if a live model writes it.
Do not score the scripted offline reply as evidence of live-model reasoning.

## Retain rules, refresh facts

A reusable lesson can tell an agent to resolve calendar dates in the application's
timezone or to inspect CI evidence before reporting success. Yesterday's commit
list belongs in the activity index and must be refreshed. It should not become
a permanent playbook fact.

For low-latency applications, use the indexed query path for supported questions
and reserve model-written summaries for requests that need them. Measure SQLite,
provider and network time separately with the example's benchmark command.
