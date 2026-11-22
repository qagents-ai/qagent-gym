# Correlated Gym and qagents traces

`TracingGenerator` and `TracingEvaluator` accept a configured OpenTelemetry tracer.
They do not set a global provider, choose an exporter or open a connection.
The tested SDK/API version is **1.45.2**.

```console
python -m pip install -e ".[tracing]"
python -m examples.otel_tracing
```

The example uses an in-memory exporter and the real qagents `OTelHooks` with
`record_content=False`. A `gym.episode` span parents generation, qagents execution
and evaluation. Its deterministic answer is `4`, reward is `1`, and `one_trace`
is true. This verifies correlation, not the quality of a live model.

```python
from qagent_gym.integrations import TracingEvaluator, TracingGenerator

def instrument(generator, evaluator, configured_tracer):
    return (
        TracingGenerator(generator, configured_tracer),
        TracingEvaluator(evaluator, configured_tracer),
    )
```

Generation spans record task ID, split, namespace, context revision, bullet and
memory counts, result status and observable step count. Evaluation spans record
task ID, evaluator ID and numeric reward. Prompts, answers and feedback are not
recorded by these wrappers. IDs and namespaces are caller-controlled: use opaque
identifiers if they would otherwise reveal sensitive information.

Errors and cancellation propagate to the caller. Wrapper spans record only the
exception class and a fixed status description; they suppress automatic exception
messages and stack events. Incomplete or paused generations retain their original
status and receive an error span status. Other instrumentation in a delegate may
have its own content policy; these wrappers do not sanitize those separate spans.

The application owns the provider/exporter and its shutdown. For external
collectors, configure authentication, sampling and retention in the host.
The [official Python instrumentation guide](https://opentelemetry.io/docs/languages/python/instrumentation/)
describes tracer/provider setup and parent context propagation.
