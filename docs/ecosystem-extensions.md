# Environment libraries and agent frameworks

Reviewed on 2026-11-19. This update extends the [earlier ecosystem survey](ecosystem.md)
with environment execution and repeated-trial reliability.

## Evidence and selection

The review inspected official repository metadata, default-branch commits, READMEs,
release identifiers and documentation for 12 candidates. The
[metadata snapshot](../assets/research/ecosystem-2026-11-19.json) records the checked
commit and README hashes. Repository links below point to those commits. A missing
release value means the latest-release endpoint returned no release; it does not
establish that a project has no tags.

This is an API and architecture review. Upstream benchmark claims were not
replicated. Native integration tests were performed for Gymnasium and LangGraph,
with QAgents as the execution runtime.

| Library | Checked source / repository license | Useful mechanism | Decision and evidence |
| --- | --- | --- | --- |
| Farama-Foundation/Gymnasium | [v1.4.0](https://github.com/Farama-Foundation/Gymnasium/blob/68c3f95def1b9d91922db5363a28dc1d4a83a960/README.md); MIT | Explicit reset/step, seeding, action spaces, and separate termination/truncation. | Implemented and exercised with the native SDK. |
| WooooDyy/AgentGym | [default branch](https://github.com/WooooDyy/AgentGym/blob/3ef9235d23e68e7c2920c5422ad957dc8ced5c6c/README.md); MIT | Separates environment servers, clients and the agent controller; exposes reset/step and observation APIs. | Adopted the interaction boundary. Its HTTP environments have not been connected. |
| R2E-Gym/R2E-Gym | [default branch](https://github.com/R2E-Gym/R2E-Gym/blob/0d94c4eb9431cd195c55a7ea3abd54006c9a1735/README.md); Apache-2.0 | Executable coding environments and independent test-based rewards. | Useful for a later coding-task backend. No R2E images or tasks were run here. |
| harbor-framework/harbor | [v0.26.0](https://github.com/harbor-framework/harbor/blob/a215bfa03ccc026d22a0f11d5d6c51a8a46f0510/README.md); Apache-2.0 | Container-backed benchmark tasks, agent adapters, and rollout collection. | Useful for isolated coding evaluations. No Harbor task or container backend was run here. |
| sierra-research/tau2-bench | [v2.0.0](https://github.com/sierra-research/tau2-bench/blob/c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4/README.md); MIT | Tool/agent/user tasks and explicit evaluation criteria separate from generated answers. | Informed verification boundaries; repeated success metrics also cite the original tau-bench paper. |
| microsoft/agent-lightning | [v1.0.3](https://github.com/microsoft/agent-lightning/blob/2070bad5113e52f1aa2e5bdba822d4b24b197c05/README.md); MIT | Agent execution and RL optimization are separated; current training examples use a dedicated training stack. | Keep the existing reward bridge. New observable trajectories add evidence, while sampled token traces and optimizer runs remain separate work. |
| langchain-ai/langgraph | [1.2.23](https://github.com/langchain-ai/langgraph/blob/7830b91a24e657a8167e2dbf40c3fa5a77ef5d93/README.md); MIT | Explicit graph state, persistence and configurable execution. | Existing generator now drives multi-step environments. Native rollout replay was tested. |
| Microsoft Agent Framework | Repository API returned HTTP 404; [official checkpoint docs](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints?pivots=programming-language-python) were reviewed. | Explicit workflow checkpoints and resume. | Source version and native compatibility remain unverified. |
| openai/openai-agents-python | [v0.25.0](https://github.com/openai/openai-agents-python/blob/91fb508a8931016efa793b033147d08cd20308cf/README.md); MIT | Agents, handoffs, guardrails, sessions and tracing. | QAgents already provides corresponding execution seams. No new runtime dependency or SDK adapter was added. |
| pydantic/pydantic-ai | [v2.71.0](https://github.com/pydantic/pydantic-ai/blob/e272a43519b70d29422540836cd711ab802f4e69/README.md); MIT | Typed agents, validation, durable execution options and separate evaluation tooling. | Supports retaining typed boundaries and independent evaluation. No native Pydantic AI agent run was performed. |
| gepa-ai/gepa | [v0.1.4](https://github.com/gepa-ai/gepa/blob/ce5b0d6e3d6af6c6dbec6de2c54536bbb961d309/README.md); MIT | Feedback-driven optimization with candidate evaluation and reflective data. | Existing reflection exports remain available. Keep optimization validation separate from the final held-out test set. |
| OpenPipe/ART | [v0.5.24](https://github.com/OpenPipe/ART/blob/55b423c3004afeec3362fd35717b47356dfdd9fa/README.md); Apache-2.0 | RL client/server architecture around rollouts, environment rewards and GRPO training. | A possible external training backend. No ART service, GPU training or weight update was run. |

The original [tau-bench paper](https://arxiv.org/abs/2406.12045) motivates measuring
repeatability across attempts and verifying the resulting application state.
Its pass^k question differs from pass@k: every attempt must work, rather than at
least one attempt. The implemented estimators and their assumptions are documented
in the [QAgents reliability guide](https://github.com/qagents-ai/qagents/blob/main/docs/guides/reliability.md).

## Implemented extensions

1. **QAgents trial factories.** `TrialTarget` receives the case and repetition
   index. A caller can select a seed and create isolated fixtures without shared
   counters or modifying dataset metadata.
2. **QAgents reliability reports.** Pass@k and pass^k use combinatorial estimators.
   Paired case bootstrap intervals retain all repetitions of a task in one
   resampling unit and reject mismatched contracts and incomplete trial sets.
3. **Gym environment rollouts.** A bounded reset/act/step loop records immutable
   action and observation transitions, environment rewards, seed and stop reason.
   Native termination, environment truncation, controller limits and incomplete
   policies remain distinguishable.
4. **Gymnasium integration.** An injected environment is owned by one rollout.
   Calls run serially on a dedicated worker thread; action/observation spaces and
   modern five-value step results are validated.
5. **Context learning through an environment.** `RolloutGenerator` connects the
   environment to existing Gym training and frozen evaluation. Diagnostic info
   stays in audit evidence; the generation output is the public final observation.
6. **Native replay evaluation.** The same decision policy runs through QAgents
   and LangGraph on deterministic and stochastic FrozenLake tasks. Records compare
   actual observations, actions, rewards and end flags.

See [the rollout guide](environment-rollouts.md) for commands, examples and the
recorded evaluation. Optional frameworks remain outside the base import path.

The [verification record](evaluation-2026-11-19.md) lists full suites, installed
package checks, recovery tests and the hosted CI limitation.

## Further integration choices

For coding tasks, a container-backed Harbor or R2E environment is the next useful
backend. It needs explicit task/image versions, resource limits, independent
verifiers and a selected local or hosted executor.

For weight training, select an external trainer such as Agent Lightning or ART
after choosing the trainable model, sampled-token capture and reward contract.
The new rollout records contain observable environment interaction; they do not
supply token IDs, policy log probabilities or an optimizer. Reward/data exports
and actual model training have different acceptance criteria.

For more agent SDKs, add an adapter only when there is a concrete application
whose lifecycle and outcomes can be tested. QAgents already supports typed
tools, approvals, session state, sub-agents, handoffs, tracing and durable
workflows. The selected extensions concentrate on environment interaction and
repeatable evidence, which were missing from the reviewed local APIs.
