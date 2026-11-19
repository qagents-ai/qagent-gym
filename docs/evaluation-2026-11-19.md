# Verification of the environment and reliability extensions

Checked on 2026-11-19 on Windows 11, with CPython 3.11.18 and 3.13.18.
The [machine-readable record](../assets/research/verification-2026-11-19.json)
retains commands, log hashes, source commits and verification limits.

| Check | Result |
| --- | --- |
| QAgents full suite, Python 3.13 | 3,905 passed; 158 skipped |
| QAgents full suite, fresh Python 3.11 environment | 3,870 passed; 162 skipped |
| Opt-in Effects mutation suite | 68 passed; 11 deselected |
| Opt-in process termination and team recovery suite | 26 passed |
| Gym full source suite, native Gymnasium and LangGraph installed | 370 passed; 15 skipped |
| Gym full suite with both libraries loaded from installed wheels | 370 passed; 15 skipped |
| Installed QAgents wheel: evaluation, dispatch, crash clock and workflow replay | 225 passed |

Both wheel and source distributions build. Lint, formatting, QAgents API code
checks and strict guide builds pass. The new FrozenLake guide snippet executes
six transitions and returns reward 1 with termination true and truncation false.
Skipped tests remain outside the indicated run; they are not successful checks.
The full QAgents runs each emit one existing MLflow SQLAlchemy deprecation warning.

## Native integration evidence

The [environment replay](environment-rollouts.md#native-replay-benchmark) runs
80 Gymnasium episodes through native QAgents and LangGraph execution. Matching
task/seed pairs have identical observations, actions, rewards, flags and end reasons.
Each framework records 38 terminations and two time-limit truncations.
Policies are scripted and no model provider is contacted. These results verify
adapter behavior; they do not demonstrate better model quality or RL weight updates.

The [context-learning example](https://github.com/mpvasilis/qagent-gym/blob/main/examples/environment_learning.py)
checks failed practice, a stored procedure and successful held-out episodes while
memory and playbook stay frozen.

## Recovery and review findings

The Python 3.11 run exposed coarse Windows timer behavior in existing main.
Dispatch now refuses a callback after its timeout decision even if the clock
still reads before the deadline. The virtual crash loop uses a positive
nanosecond coalescing epsilon instead of the host's 15.6 ms quantum. Regression
tests for both behaviors fail on original main and pass with the fixes.

Workflow concurrency and cancellation tests now synchronize on actual entry,
rather than assuming callbacks start within a few milliseconds. Existing bounds
and outcome assertions are retained. Review probes deliberately broke pass@k
arithmetic and duplicate-key detection; the corresponding tests rejected each
mutation. These probes are separate from the 68-test Effects mutation suite.

## Hosted CI and remaining limits

The [GitHub Actions run](https://github.com/mpvasilis/qagents/actions/runs/44525958295)
could not start any job because the repository's Actions budget prevents use.
Every job executed zero steps. The table above records local verification and
does not claim hosted CI passed. Billing and repository protection settings were
left unchanged.

No Linux runner or PostgreSQL service was exercised locally. Docker was installed
but its daemon was unavailable. Provider sandbox calls and GPU/model training
were outside this check. The [ecosystem review](ecosystem-extensions.md) separates
implemented native integrations from libraries reviewed for future adoption.
