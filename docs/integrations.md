# Six integration recipes

These adapters extend the Gym while leaving service clients and framework
configuration with the application. Base installation imports no optional SDK.
Run example modules from a source checkout; the wheel contains the adapters.

| Integration | Install extra | Example command | Observed boundary |
| --- | --- | --- | --- |
| Mem0 | `.[mem0]` | `python -m examples.mem0_memory` | Scoped memory and a read-only snapshot; default uses a fixture with real qagents tools. |
| LangGraph | `.[langgraph]` | `python -m examples.langgraph_agent` | Actual graph execution, fresh checkpoint threads and preserved approval pauses. |
| DSPy | `.[dspy]` | `python -m examples.dspy_agent` | Actual asynchronous Module calls with a deterministic local policy. |
| ElevenLabs | `.[elevenlabs]` | `python -m examples.elevenlabs_voice --language el` | Bounded streamed PCM converted to WAV; default fixture produces silence. |
| Twilio | `.[twilio]` | `python -m examples.twilio_voice` | Official signed request validation and generated TwiML; no phone call. |
| OpenTelemetry | `.[tracing]` | `python -m examples.otel_tracing` | Native SDK spans connect a Gym episode to qagents execution and reward. |

See the [memory/framework guide](integrations-memory-frameworks.md),
[voice service guide](integrations-voice-services.md), and
[tracing guide](integrations-tracing.md) for tested versions, APIs and limits.

The existing [Inspect and GEPA exports](optimization.md) separate answer keys and
training feedback. [Lightning and policy training](rl.md) remain transport/export
boundaries: these recipes do not train model weights or establish benchmark gains.

Use [background jobs](background-jobs.md) for persisted run scheduling and
[application scenarios](examples.md) for KA220 drafts, a fictional banking ledger
and English/Greek local audio. Choose one integration and measure its contribution
on fixed held-out tasks before combining several.
