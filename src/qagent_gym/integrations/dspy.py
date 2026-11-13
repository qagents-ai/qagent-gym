"""Invoke an explicitly supplied DSPy program without configuring a model or optimizer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol

from qagent_gym.records import Context, Generation, Step, Task


class AsyncProgram(Protocol):
    async def acall(self, *, question: str, context: str) -> object: ...


class DSPyGenerator:
    """Call program.acall(question=..., context=...) and validate Prediction.toDict().

    Modules should implement aforward, or otherwise support the DSPy acall API.
    Exceptions propagate. No model is configured, no optimizer is invoked and no
    learned program state is changed by this adapter. Caller-owned programs may
    have side effects; Gym's evaluation freeze covers the injected context only.
    """

    def __init__(
        self, program: AsyncProgram, *, output_field: str = "answer", max_context_chars: int = 20000
    ) -> None:
        if not callable(getattr(program, "acall", None)):
            raise ValueError("DSPy program must expose async acall")
        if not isinstance(output_field, str) or not output_field.strip():
            raise ValueError("output_field must be a nonempty string")
        if type(max_context_chars) is not int or max_context_chars < 1:
            raise ValueError("max_context_chars must be a positive integer")
        self.program = program
        self.output_field = output_field
        self.max_context_chars = max_context_chars

    async def generate(self, task: Task, context: Context) -> Generation:
        rendered = context.render()
        if len(rendered) > self.max_context_chars:
            raise ValueError("context exceeds max_context_chars")
        # Namespace and split are explicit context data, not answer keys or program configuration.
        payload = json.dumps(
            {"namespace": context.namespace, "split": task.split, "learned_context": rendered},
            ensure_ascii=False,
            sort_keys=True,
        )
        prediction = await self.program.acall(question=task.prompt, context=payload)
        convert = getattr(prediction, "toDict", None)
        if not callable(convert):
            raise ValueError("DSPy program must return a Prediction with toDict()")
        values = convert()
        if not isinstance(values, Mapping):
            raise ValueError("DSPy Prediction.toDict() must return a mapping")
        answer = values.get(self.output_field)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"DSPy output {self.output_field!r} must be a nonempty string")
        return Generation(
            answer,
            steps=(
                Step("message", payload, role="system"),
                Step("message", task.prompt, role="user"),
                Step("message", answer, role="assistant"),
            ),
            model="dspy",
        )
