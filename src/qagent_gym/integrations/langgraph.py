"""Run an injected compiled LangGraph with a fresh checkpoint thread per episode."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from typing import Any, Protocol

from qagent_gym.records import Context, Generation, Step, Task


class CompiledGraph(Protocol):
    checkpointer: object

    async def ainvoke(self, input: dict, config: dict, **kwargs: Any) -> object: ...

    async def aget_state(self, config: dict) -> object: ...


class LangGraphGenerator:
    """Use the public v2 invocation contract (LangGraph 1.2.21).

    Input fields are question, context, namespace, task_id and split. The graph
    must return a mapping with a nonempty string at output_key. Each call gets
    a new thread; this adapter never resumes a paused graph. A caller-owned
    checkpointer may retain those runs for inspection, not automatic replay.

    Gym freezes its supplied context during evaluation. Graph nodes remain
    responsible for any external stores, model state or side effects they use.
    """

    def __init__(
        self,
        graph: CompiledGraph,
        *,
        output_key: str = "output",
        recursion_limit: int = 25,
        max_context_chars: int = 20000,
    ) -> None:
        if not callable(getattr(graph, "ainvoke", None)) or not hasattr(graph, "checkpointer"):
            raise ValueError("graph must be compiled and expose ainvoke and checkpointer")
        if graph.checkpointer and not callable(getattr(graph, "aget_state", None)):
            raise ValueError("checkpointed graph must expose aget_state for pause detection")
        if not isinstance(output_key, str) or not output_key.strip():
            raise ValueError("output_key must be a nonempty string")
        for label, value in (
            ("recursion_limit", recursion_limit),
            ("max_context_chars", max_context_chars),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.graph = graph
        self.output_key = output_key
        self.recursion_limit = recursion_limit
        self.max_context_chars = max_context_chars

    async def generate(self, task: Task, context: Context) -> Generation:
        rendered = context.render()
        if len(rendered) > self.max_context_chars:
            raise ValueError("context exceeds max_context_chars")
        namespace_key = hashlib.sha256(context.namespace.encode("utf-8")).hexdigest()[:24]
        thread_id = f"qagent-gym:{namespace_key}:{uuid.uuid4().hex}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.recursion_limit,
            "metadata": {
                "gym_namespace": context.namespace,
                "gym_task_id": task.task_id,
                "gym_split": task.split,
            },
        }
        inputs = {
            "question": task.prompt,
            "context": rendered,
            "namespace": context.namespace,
            "task_id": task.task_id,
            "split": task.split,
        }
        result = await self.graph.ainvoke(inputs, config=config, version="v2", stream_mode="values")
        if not hasattr(result, "value") or not hasattr(result, "interrupts"):
            raise ValueError("LangGraph must return v2 GraphOutput with value and interrupts")
        interrupts = result.interrupts
        if not isinstance(interrupts, (list, tuple)):
            raise ValueError("LangGraph interrupts must be a sequence")
        steps = (
            Step("message", json.dumps({"thread_id": thread_id}, sort_keys=True), role="system"),
            Step("message", json.dumps(inputs, ensure_ascii=False, sort_keys=True), role="user"),
        )
        pending = bool(interrupts)
        if self.graph.checkpointer:
            snapshot = await self.graph.aget_state(config)
            next_nodes = getattr(snapshot, "next", None)
            if not isinstance(next_nodes, tuple):
                raise ValueError("LangGraph checkpoint must expose a tuple of next nodes")
            pending = pending or bool(next_nodes)
        if pending:
            return Generation(
                "", status="paused", steps=steps, model="langgraph", stop_reason="graph_interrupted"
            )
        value = result.value
        if not isinstance(value, Mapping):
            raise ValueError("LangGraph final state must be a mapping")
        answer = value.get(self.output_key)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"LangGraph output {self.output_key!r} must be a nonempty string")
        return Generation(
            answer, steps=steps + (Step("message", answer, role="assistant"),), model="langgraph"
        )
