"""Correlate real qagents spans with Gym context and reward in a local collector.

Run: python -m examples.otel_tracing
Requires the optional tracing extra; no network exporter or global provider.
"""

import asyncio
import json

from qagents import Agent
from qagents.models.testing import FunctionModel
from qagents.tracing import OTelHooks

from qagent_gym.integrations.otel import TracingEvaluator, TracingGenerator
from qagent_gym.qagents_adapter import QAgentsGenerator
from qagent_gym.records import Context, Reward, Task


async def run():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("quentin.gym.example")
    agent = Agent(
        FunctionModel(lambda request: "4"),
        hooks=[OTelHooks(tracer, record_content=False)],
        name="traced-quentin",
    )

    class Exact:
        async def evaluate(self, task, generation):
            return Reward(float(generation.output == "4"), "exact-match", "Checked locally.")

    try:
        with tracer.start_as_current_span("gym.episode"):
            task = Task("math-check", "What is 2 + 2?", "eval")
            generation = await TracingGenerator(QAgentsGenerator(agent), tracer).generate(
                task, Context("tracing-demo")
            )
            reward = await TracingEvaluator(Exact(), tracer).evaluate(task, generation)
        spans = exporter.get_finished_spans()
        return {
            "mode": "native OpenTelemetry and qagents; in-memory exporter",
            "output": generation.output,
            "reward": reward.value,
            "one_trace": len({span.context.trace_id for span in spans}) == 1,
            "spans": [
                {
                    "name": span.name,
                    "parented": span.parent is not None,
                    "attributes": dict(span.attributes),
                    "status": span.status.status_code.name,
                }
                for span in spans
            ],
        }
    finally:
        provider.shutdown()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), indent=2))
