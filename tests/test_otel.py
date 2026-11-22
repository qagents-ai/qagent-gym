import asyncio
import json

import pytest

from qagent_gym.integrations.otel import TracingEvaluator, TracingGenerator
from qagent_gym.records import Context, Generation, Reward, Task

pytest.importorskip("opentelemetry.sdk.trace")
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("test.gym"), exporter
    provider.shutdown()


def test_real_qagents_spans_share_parent_context():
    from examples.otel_tracing import run

    report = asyncio.run(run())
    assert report["one_trace"] and report["reward"] == 1
    assert {"gym.episode", "qagent_gym.generate", "qagent_gym.evaluate"} <= {
        span["name"] for span in report["spans"]
    }
    assert any(span["name"].startswith("invoke_agent") for span in report["spans"])
    assert all(span["parented"] for span in report["spans"] if span["name"] != "gym.episode")
    assert "What is 2 + 2?" not in json.dumps(report["spans"])


def test_success_preserves_output_and_records_only_selected_metadata(tracing):
    tracer, exporter = tracing
    result = Generation("private-output")

    class Generator:
        async def generate(self, task, context):
            return result

    returned = asyncio.run(
        TracingGenerator(Generator(), tracer).generate(
            Task("opaque-task", "private-prompt"), Context("opaque-tenant")
        )
    )
    assert returned is result
    (span,) = exporter.get_finished_spans()
    assert span.attributes["qagent_gym.namespace"] == "opaque-tenant"
    assert span.attributes["qagent_gym.status"] == "completed"
    assert span.status.status_code is StatusCode.OK
    assert "private-" not in json.dumps(dict(span.attributes))


@pytest.mark.parametrize(
    "error", [ValueError("private-error"), asyncio.CancelledError("private-error")]
)
def test_errors_propagate_without_recording_message_or_stack(tracing, error):
    tracer, exporter = tracing

    class Generator:
        async def generate(self, task, context):
            raise error

    with pytest.raises(type(error)):
        asyncio.run(
            TracingGenerator(Generator(), tracer).generate(Task("id", "secret"), Context("ns"))
        )
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == type(error).__name__
    assert not span.events
    assert "private-error" not in str(span.status) + str(span.attributes)


def test_incomplete_generation_remains_incomplete(tracing):
    tracer, exporter = tracing

    class Generator:
        async def generate(self, task, context):
            return Generation("", status="paused", stop_reason="approval")

    result = asyncio.run(
        TracingGenerator(Generator(), tracer).generate(Task("id", "input"), Context("ns"))
    )
    assert result.status == "paused"
    assert exporter.get_finished_spans()[0].status.status_code is StatusCode.ERROR


def test_evaluation_logs_number_without_feedback(tracing):
    tracer, exporter = tracing
    expected = Reward(0.75, "evaluator-v1", "private-feedback")

    class Evaluator:
        async def evaluate(self, task, generation):
            return expected

    result = asyncio.run(
        TracingEvaluator(Evaluator(), tracer).evaluate(Task("id", "input"), Generation("answer"))
    )
    assert result is expected
    (span,) = exporter.get_finished_spans()
    assert span.attributes["qagent_gym.reward"] == 0.75
    assert "private-feedback" not in str(span.attributes)


def test_bad_delegate_result_fails_and_closes_span(tracing):
    tracer, exporter = tracing

    class Generator:
        async def generate(self, task, context):
            return "wrong contract"

    with pytest.raises(TypeError, match="Generation"):
        asyncio.run(
            TracingGenerator(Generator(), tracer).generate(Task("id", "input"), Context("ns"))
        )
    assert exporter.get_finished_spans()[0].status.status_code is StatusCode.ERROR
