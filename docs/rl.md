# Reinforcement-learning integration boundary

Gym currently runs and evaluates agents, updates context/playbooks, and exports
episode records. These operations do not update model weights. An exported JSONL
file is training data, not a completed reinforcement-learning optimizer run.
No model training, weight download, or live training service has been exercised.

## Agent Lightning v1 reward bridge

Agent Lightning v1 uses an OpenAI-compatible Gateway, a Rollout Controller, and a
trainer built on verl. The agent keeps its execution loop while the Gateway
collects model calls. Pre-v1 `LitAgent`, `Trainer`, and `emit_reward` tutorials
describe the legacy architecture. See the [v1 architecture](https://microsoft.github.io/agent-lightning/stable/05-basics/)
and [version notice](https://microsoft.github.io/agent-lightning/stable/).

Gym supplies two dependency-free functions:

```python
from qagent_gym.integrations import reward_payload, send_reward
from qagent_gym.records import Reward

reward = Reward(value=1.0, evaluator="exact-v1", feedback="Expected answer matched.")
payload = reward_payload(reward)
# {"event_type": "reward", "data": {
#     "value": 1.0, "source": "exact-v1", "reason": "Expected answer matched."
# }}
```

`send_reward(reward, event_url=..., api_key=..., transport=...)` calls the supplied
synchronous transport exactly once as
`transport(event_url, json=payload, headers={"Authorization": "Bearer ..."})`.
The transport must check the response and raise on failure or an unknown outcome.
Gym has no default HTTP implementation and performs no retry. `reward_payload`
rejects non-finite values. The evaluator becomes the event's `source`; feedback
becomes `reason`, following the [official reward event shape](https://github.com/microsoft/agent-lightning/blob/main/examples/search_r1/agents/search_r1_agent.py).

For a future full adapter, the Controller supplies `AGL_OPENAI_BASE_URL`,
`AGL_EVENT_URL`, and `AGL_KEY`. A local agent class exposes `async run()`; connect
the qagents model client to the supplied proxy and send the terminal reward.
The v1 trainer entry point is `agentlightning.verl.entrypoint.run_ppo`, with
JSON-like training and validation sequences. Its trace aggregator merges model
calls only where token-prefix continuity holds and masks tool observations from
the policy loss. Text-only Gym exports cannot reconstruct missing sampled token
IDs or log probabilities. See [trainer configuration](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/).

The current [installation guide](https://microsoft.github.io/agent-lightning/stable/00-installation/)
tests `verl==0.8.0` with `vllm==0.20.2` and CUDA 13.0, or `verl==0.7.1` with
`vllm==0.12.0` and CUDA 12.9, using `flash-attn==2.8.3`. The
[local controller requires Linux/WSL rather than native Windows](https://microsoft.github.io/agent-lightning/stable/30-controller-configuration/).
The quickstart uses an A100 and defaults to W&B uploads. Keep this stack separate
from Gym's base environment; pin a tested source revision before using it, since
the current `/stable/` documentation also displays a development warning.

## Alternative training backends

[TRL's GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) accepts an
experimental `rollout_func(prompts, trainer)`. It must generate the required
number of completions per prompt and return `prompt_ids`, `completion_ids`, and
`logprobs`; optional `logprob_token_ids` describe sampling support. Additional
fields reach the reward functions. Its experimental `environment_factory`
interface instead exposes tools plus `reset()` and optional `get_reward()` while
TRL owns generation. The [release page](https://github.com/huggingface/trl/releases)
listed v1.17.1 on 2026-11-13; rolling documentation is not a tested compatibility
promise. Gym does not implement either training adapter yet.

Direct [verl AgentLoopBase](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
requires `async run(sampling_params, **kwargs) -> AgentLoopOutput`, with prompt
and response token IDs and a response mask distinguishing generated tokens from
tool output. Its documentation labels this API alpha and subject to change.
Prefer the gateway route when preserving qagents' existing agent loop matters.

A real RL experiment still needs a trainable checkpoint, tokenizer, task split,
reward, sampled-token traces, and an optimizer backend. Freeze memory/playbook
context when comparing policies and keep evaluation tasks outside adaptation.
The current fake-transport tests verify only reward payloads, validation, and
error propagation; they do not establish end-to-end training compatibility.

Agent Lightning uses [MIT](https://github.com/microsoft/agent-lightning/blob/main/LICENSE);
TRL and verl use [Apache-2.0](https://github.com/huggingface/trl/blob/main/LICENSE)
([verl license](https://github.com/verl-project/verl/blob/main/LICENSE)). Selected
model weights and datasets have their own licenses.
