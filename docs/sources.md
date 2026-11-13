# Sources and license notes

Reviewed 2026-11-13. Version values below are source/release snapshots, not a
guarantee that an unpinned future installation will expose the same API.

| Component | Reference | License / compatibility |
| --- | --- | --- |
| ACE paper | [Agentic Context Engineering, v3](https://arxiv.org/abs/2510.04618v3) | Describes context adaptation, not model-weight updates. |
| Authors' ACE code | [ace-agent/ace](https://github.com/ace-agent/ace), [metadata](https://github.com/ace-agent/ace/blob/main/pyproject.toml), [license](https://github.com/ace-agent/ace/blob/main/LICENSE.txt) | Apache-2.0; source version 0.1.0, Python >=3.10. The project website's MIT footer does not match the repository license. |
| Community ACE engine | [Kayba](https://github.com/kayba-ai/agentic-context-engine), [metadata](https://github.com/kayba-ai/agentic-context-engine/blob/main/pyproject.toml) | Apache-2.0; ace-framework 0.13.0, Python >=3.12. Separate implementation; both import as ace. |
| Mem0 | [release history](https://github.com/mem0ai/mem0/releases), [license](https://github.com/mem0ai/mem0/blob/main/LICENSE) | Apache-2.0; 2.2.1 contract targeted. |
| Agent Lightning | [docs](https://microsoft.github.io/agent-lightning/stable/), [license](https://github.com/microsoft/agent-lightning/blob/main/LICENSE) | MIT; v1 differs substantially from 0.x examples. Pin a tested commit for a real integration. |
| TRL | [GRPO API](https://huggingface.co/docs/trl/grpo_trainer), [license](https://github.com/huggingface/trl/blob/main/LICENSE) | Apache-2.0; custom rollout API is experimental. |
| verl | [agent-loop API](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), [license](https://github.com/verl-project/verl/blob/main/LICENSE) | Apache-2.0; direct agent-loop integration is deferred. |
| LangMem (not implemented) | [standalone example](https://github.com/langchain-ai/langmem/blob/main/examples/standalone_examples/custom_store_example.py), [license](https://github.com/langchain-ai/langmem/blob/main/LICENSE) | MIT; optional future alternative when LangGraph is already used. |

Gym implements its own small learning loop; it does not vendor these projects'
code or prompts. Preserve applicable license/notice files if adding copied code
later. Framework licenses do not establish the rights to model weights, datasets,
or exported task content. qagents remains a separate dependency.
