# English and Greek voice agents

The runnable example connects local speech recognition, a qagents Agent, and local
speech synthesis. It accepts a WAV file and produces sentence-sized response WAVs.
Both English and Greek were exercised with real ASR/TTS on this Windows machine.
The qagents response policy is deterministic Python. General language-model quality
is not measured by this example.

## Run the native example

From the repository root with qagent-gym and qagents installed, create a separate
speech environment. These packages are optional and are not added to Gym's base
dependencies or the qagents environment.

```powershell
python -m venv .work/voice-venv
.work/voice-venv/Scripts/python -m pip install -r examples/voice-requirements.txt
$env:HF_HOME = Join-Path (Get-Location) '.work/hf-voice-cache'
.work/voice-venv/Scripts/python -m examples.download_voice_model --output .work/voice-model
python -m examples.voice_agent --asr-python .work/voice-venv/Scripts/python.exe --model .work/voice-model --output .gym/voice --repeats 3
```

The download command explicitly fetches the public
[Systran faster-whisper tiny model](https://huggingface.co/Systran/faster-whisper-tiny),
revision `d90ca5fe260221311c53c58e660288d3deb8d356`. Inference loads local files only.
Model weights and environments stay out of Git. Native TTS requires Windows
System.Speech and installed, enabled `en-US` and `el-GR` voices. The program selects
an exact matching culture and fails if it is missing. It uses an existing `pwsh`
when available, otherwise Windows PowerShell. It does not change execution policies.

The tested speech stack is faster-whisper 1.2.1, CTranslate2 4.9.0 and PyAV 18.1.0,
with CPU int8 inference and four threads. A fresh unpinned install initially chose
PyAV 19.0.2 and failed in audio decoding. The pinned requirements resolve the
[upstream PyAV API incompatibility](https://github.com/SYSTRAN/faster-whisper/pull/1495).

## Observed local results

On 2026-11-13, Python 3.13.18 / Windows / AMD Ryzen 7 5700, three trials per
language used one synthetic input phrase per language:

| Language | Median first response WAV | Nearest-rank p95 | Word error rate |
| --- | ---: | ---: | ---: |
| English | 483 ms | 507 ms | 0 on all three trials |
| Greek | 555 ms | 559 ms | 0 on all three trials |

English input: “Hello, I need help with a project.” Greek input:
“Γεια σου. Θέλω βοήθεια με ένα πρόγραμμα.” Native voices were Microsoft Zira
Desktop (`en-US`) and Microsoft Stefanos (`el-GR`). Speech-worker startup took
1.69 seconds and initial qagents setup took 0.56 seconds, reported separately.

The timing begins when the input WAV is ready, includes validation, recognition,
the scripted qagents response, first-sentence synthesis and its file write, and
ends when the first response WAV is ready. It excludes input recording, playback,
microphone endpointing, networking and a real LLM provider. Three repeats of two
clean synthetic utterances are a smoke benchmark, not a general latency or ASR
accuracy claim. The small-sample p95 equals the slowest trial. Changing hardware,
voice, language content, noise, model or concurrency can change these results.

## Reuse the pipeline

`VoicePipeline` accepts async `transcribe(path, language)`,
`respond(text, language)` and `synthesize(text, language)` callables. The latter
returns WAV bytes. The supplied responder calls a real qagents Agent with finite
limits and a FunctionModel policy. Replace that boundary with your configured
agent while preserving its completion/status checks.

`run(..., on_audio=callback)` delivers each validated WAV path before synthesizing
the next sentence. Each chunk is an independent WAV: decode or play each chunk,
or concatenate its PCM frames rather than concatenating WAV file bytes. The full
agent response is available before TTS starts. Model-token streaming is not
implemented. The native demo does not record a microphone or play sound.

`LatestTurn` cancels and awaits the previous turn before starting another. A
cancelled or failed turn removes only the files it created. Applications still
need to stop their playback queue when a turn is interrupted. Already played
audio cannot be recalled. Native subprocess cancellation closes that worker to
prevent consuming a stale reply. Restart it before the next native request.

Inputs are limited to 30 seconds, mono/stereo PCM, 8 to 48 kHz, 1 to 4 bytes/sample and
12 MB of declared PCM. Header checks precede the bulk read. Empty recognition,
oversized replies, unavailable voices and incomplete agent runs fail explicitly.
The pipeline has a 15-second deadline. Each native worker request also has its
own deadline, including test-fixture synthesis. Startup is bounded separately.

## Verification and next integrations

```powershell
python -m pytest tests/test_voice_agent.py -q
```

Fifteen tests cover both language responses through qagents, real callback order,
blank recognition, input bounds, timeouts, interruption, cancelled replacement,
partial-write cleanup and worker teardown. Speech providers in these unit tests
are fakes. The native six-trial run above exercised the installed speech stack.
Review found and regression-tested cancellation swallowing and partial file
cleanup defects before they were fixed.

For a live browser/phone application, evaluate
[LiveKit Agents](https://docs.livekit.io/agents/) for WebRTC, turn detection and
interruption handling, or [Pipecat](https://github.com/pipecat-ai/pipecat) for
composable realtime audio pipelines and transports. Keep qagents as the tool and
permission execution boundary. Neither framework is integrated in this example.
