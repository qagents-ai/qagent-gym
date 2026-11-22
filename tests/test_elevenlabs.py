import asyncio
import io
import wave
from types import SimpleNamespace

import pytest

from qagent_gym.integrations.elevenlabs import ElevenLabsSynthesizer, VoiceSynthesisError


class Stream:
    def __init__(self, chunks=(b"\x01\x00", b"\x02\x00"), *, wait=False):
        self.chunks = iter(chunks)
        self.closed = False
        self.wait = wait
        self.entered = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.entered.set()
        if self.wait:
            await asyncio.Event().wait()
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


def synthesizer(stream, **kwargs):
    calls = []

    def create(**request):
        calls.append(request)
        return stream

    client = SimpleNamespace(text_to_speech=SimpleNamespace(stream=create))
    return ElevenLabsSynthesizer(client, voice_id="fixture", **kwargs), calls


def test_pcm_chunks_form_a_real_wav_and_language_is_explicit():
    stream = Stream((b"\x01", b"\x00\x02", b"\x00"))
    synth, calls = synthesizer(stream)
    result = asyncio.run(synth("Γεια σας", "el"))
    with wave.open(io.BytesIO(result), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        assert wav.readframes(2) == b"\x01\x00\x02\x00"
    assert stream.closed
    assert calls[0]["language_code"] == "el"
    assert calls[0]["output_format"] == "pcm_24000"
    assert calls[0]["request_options"]["max_retries"] == 0


@pytest.mark.parametrize("chunks", [(), (b"a",), ("not bytes",), (b"RIFFabcd",), (b"x" * 48002,)])
def test_invalid_or_oversized_audio_is_closed_and_rejected(chunks):
    stream = Stream(chunks)
    synth, _ = synthesizer(stream, max_audio_seconds=1)
    with pytest.raises(VoiceSynthesisError):
        asyncio.run(synth("Hello", "en"))
    assert stream.closed


@pytest.mark.parametrize(
    "text,language", [("", "en"), ("a" * 601, "en"), ("Hi\x00", "en"), ("Hi", "fr")]
)
def test_invalid_text_and_language_never_call_provider(text, language):
    synth, calls = synthesizer(Stream())
    with pytest.raises(ValueError):
        asyncio.run(synth(text, language))
    assert not calls


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_deadline(value):
    with pytest.raises(ValueError):
        synthesizer(Stream(), timeout_s=value)


def test_timeout_closes_provider_stream():
    stream = Stream(wait=True)
    synth, _ = synthesizer(stream, timeout_s=0.01)
    with pytest.raises(TimeoutError):
        asyncio.run(synth("Hello", "en"))
    assert stream.closed


def test_external_cancellation_propagates_and_closes_stream():
    async def scenario():
        stream = Stream(wait=True)
        synth, _ = synthesizer(stream)
        task = asyncio.create_task(synth("Hello", "en"))
        await stream.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed

    asyncio.run(scenario())


def test_cleanup_failure_does_not_replace_caller_cancellation():
    class BrokenClose(Stream):
        async def aclose(self):
            self.closed = True
            raise RuntimeError("private transport details")

    async def scenario():
        stream = BrokenClose(wait=True)
        synth, _ = synthesizer(stream)
        task = asyncio.create_task(synth("Hello", "en"))
        await stream.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed

    asyncio.run(scenario())


def test_cleanup_failure_after_success_is_sanitized():
    class BrokenClose(Stream):
        async def aclose(self):
            raise RuntimeError("private transport details")

    synth, _ = synthesizer(BrokenClose())
    with pytest.raises(VoiceSynthesisError, match="could not be closed"):
        asyncio.run(synth("Hello", "en"))


def test_provider_error_is_sanitized():
    class Broken(Stream):
        async def __anext__(self):
            raise RuntimeError("secret-api-key=do-not-leak")

    stream = Broken()
    synth, _ = synthesizer(stream)
    with pytest.raises(VoiceSynthesisError, match="speech provider request failed") as error:
        asyncio.run(synth("Hello", "en"))
    assert "secret" not in str(error.value)
    assert error.value.__suppress_context__
    assert stream.closed


def test_offline_demo_executes_real_qagents_tool_and_pipeline(tmp_path):
    from examples.elevenlabs_voice import run_demo
    from examples.voice_agent import validate_wav

    report = asyncio.run(run_demo(tmp_path, "el"))
    assert report["tool_executions"] == ["el"]
    assert report["closed_streams"] == report["provider_requests"] == 1
    assert report["quality_or_service_latency_measured"] is False
    assert report["pipeline"]["reply"] == "Πώς μπορούμε να βοηθήσουμε;"
    assert validate_wav(report["pipeline"]["audio_files"][0]) == 0.1


def test_live_entrypoint_requires_credentials_before_optional_sdk(tmp_path, monkeypatch):
    from examples.elevenlabs_voice import live_speech

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="must be configured"):
        asyncio.run(live_speech("Hello", "en", "fixture", tmp_path))
    assert list(tmp_path.iterdir()) == []
