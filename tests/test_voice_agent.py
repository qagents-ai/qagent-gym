import asyncio
import io
import struct
import sys
import wave
from pathlib import Path

import pytest

from examples.voice_agent import (
    JsonWorker,
    LatestTurn,
    VoicePipeline,
    scripted_responder,
    validate_wav,
    word_error_rate,
)


def pcm(seconds=0.1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * int(16000 * seconds))
    return stream.getvalue()


@pytest.fixture
def speech(tmp_path):
    path = tmp_path / "input.wav"
    path.write_bytes(pcm())
    return path


@pytest.mark.parametrize(
    ("language", "transcript", "expected"),
    [
        ("en", "I need help with a project.", "main goal"),
        ("el", "Θέλω βοήθεια με ένα πρόγραμμα.", "στόχος"),
    ],
)
def test_real_qagents_response_flows_to_synthesis(speech, tmp_path, language, transcript, expected):
    spoken = []

    async def transcribe(path, lang):
        assert path == speech and lang == language
        return transcript

    async def synthesize(text, lang):
        spoken.append((text, lang))
        return pcm()

    pipeline = VoicePipeline(transcribe, scripted_responder(), synthesize)
    report = asyncio.run(pipeline.run(speech, language, tmp_path / "output"))
    assert expected in report["reply"]
    assert report["transcript"] == transcript
    assert " ".join(text for text, _ in spoken) == report["reply"]
    assert all(lang == language for _, lang in spoken)
    assert report["total_ms"] >= report["first_audio_ms"] >= report["asr_ms"] >= 0
    assert all(validate_wav(Path(p)) > 0 for p in report["audio_files"])


def test_blank_recognition_never_reaches_agent_or_synthesizer(speech, tmp_path):
    async def empty(*args):
        return "   "

    async def forbidden(*args):
        pytest.fail("downstream provider called after blank recognition")

    with pytest.raises(ValueError, match="no text"):
        asyncio.run(VoicePipeline(empty, forbidden, forbidden).run(speech, "en", tmp_path))


def test_input_limits_precede_provider_calls(speech, tmp_path):
    async def forbidden(*args):
        pytest.fail("provider called on invalid input")

    pipeline = VoicePipeline(forbidden, forbidden, forbidden)
    with pytest.raises(ValueError, match="en or el"):
        asyncio.run(pipeline.run(speech, "xx", tmp_path))
    speech.write_bytes(pcm(31))
    with pytest.raises(ValueError, match="0–30"):
        asyncio.run(pipeline.run(speech, "en", tmp_path))


def test_timeout_cancels_provider(speech, tmp_path):
    cancelled = []

    async def slow(*args):
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.append(True)

    with pytest.raises(TimeoutError):
        asyncio.run(VoicePipeline(slow, slow, slow, timeout_s=0.01).run(speech, "en", tmp_path))
    assert cancelled == [True]


def test_new_turn_cancels_old_turn_without_stale_audio(speech, tmp_path):
    async def scenario():
        entered = asyncio.Event()
        calls = 0

        async def transcribe(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await asyncio.sleep(30)
            return "new request"

        async def respond(*args):
            return "New answer."

        async def synthesize(*args):
            return pcm()

        session = LatestTurn(VoicePipeline(transcribe, respond, synthesize))
        first = await session.start(speech, "en", tmp_path / "output")
        await entered.wait()
        second = await session.start(speech, "en", tmp_path / "output")
        report = await second
        assert first.cancelled()
        assert report["reply"] == "New answer."
        assert len(list((tmp_path / "output").glob("*.wav"))) == 1

    asyncio.run(scenario())


def test_cancelling_replacement_caller_does_not_start_a_new_turn(speech, tmp_path):
    async def scenario():
        entered = asyncio.Event()
        cleaning = asyncio.Event()
        runs = []

        class Pipeline:
            async def run(self, *args):
                runs.append(len(runs) + 1)
                entered.set()
                try:
                    await asyncio.sleep(30)
                finally:
                    cleaning.set()
                    if len(runs) == 1:
                        await asyncio.sleep(30)

        session = LatestTurn(Pipeline())
        first = await session.start(speech, "en", tmp_path)
        await entered.wait()
        replacement = asyncio.create_task(session.start(speech, "el", tmp_path))
        await cleaning.wait()
        replacement.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await replacement
            assert runs == [1]
        finally:
            if session.task is not first and not session.task.done():
                session.task.cancel()
            await asyncio.gather(first, session.task, return_exceptions=True)

    asyncio.run(scenario())


def test_failed_second_chunk_removes_only_this_turn_files(speech, tmp_path):
    unrelated = tmp_path / "keep.wav"
    unrelated.write_bytes(pcm())
    calls = 0

    async def transcribe(*args):
        return "hello"

    async def respond(*args):
        return "First sentence. Second sentence."

    async def synthesize(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthesis failed")
        return pcm()

    with pytest.raises(RuntimeError, match="synthesis failed"):
        asyncio.run(VoicePipeline(transcribe, respond, synthesize).run(speech, "en", tmp_path))
    assert set(tmp_path.glob("*.wav")) == {speech, unrelated}


@pytest.mark.parametrize("outer_cancellation", [True, False])
def test_worker_cancellation_closes_stream_before_reuse(tmp_path, outer_cancellation):
    worker_file = tmp_path / "worker.py"
    worker_file.write_text(
        "import json,sys,time\n"
        "print(json.dumps({'ready':True}),flush=True)\n"
        "for line in sys.stdin:\n"
        " time.sleep(30)\n"
        " print(json.dumps({'text':'late'}),flush=True)\n",
        encoding="utf-8",
    )

    async def scenario():
        request_timeout = 15 if outer_cancellation else 0.02
        async with JsonWorker(
            sys.executable, str(worker_file), request_timeout_s=request_timeout
        ) as worker:
            with pytest.raises(TimeoutError):
                if outer_cancellation:
                    await asyncio.wait_for(worker.request({"text": "hello"}), 0.02)
                else:
                    await worker.request({"text": "hello"})
            assert worker.process.returncode is not None
            with pytest.raises(RuntimeError, match="closed"):
                await worker.request({"text": "new"})

    asyncio.run(scenario())


def test_word_error_rate_counts_actual_recognition_errors():
    assert word_error_rate("Γεια σου!", "Γεια σου.") == 0
    assert word_error_rate("one two three", "one four three") == pytest.approx(1 / 3)
    assert word_error_rate("one two", "one two three") == 0.5


def test_partial_output_is_removed_if_disk_write_fails(speech, tmp_path, monkeypatch):
    original_open = Path.open

    class FailingWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def write(self, payload):
            self.stream.write(payload[:20])
            raise OSError("simulated full disk")

        def __exit__(self, *args):
            self.stream.close()

    def patched_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if mode == "xb" and path.name.startswith("en-"):
            return FailingWrite(stream)
        return stream

    async def transcribe(*args):
        return "hello"

    async def respond(*args):
        return "Hello."

    async def synthesize(*args):
        return pcm()

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(OSError, match="full disk"):
        asyncio.run(VoicePipeline(transcribe, respond, synthesize).run(speech, "en", tmp_path))
    assert list(tmp_path.glob("en-*.wav")) == []


def test_malicious_wav_header_is_rejected_before_any_bulk_read(speech, monkeypatch):
    payload = bytearray(pcm())
    struct.pack_into("<I", payload, 24, 0xFFFFFFFF)  # sample rate
    struct.pack_into("<I", payload, 40, 0xFFFFFFF0)  # declared data bytes
    speech.write_bytes(payload)

    def forbidden_read(*args):
        pytest.fail("unbounded PCM read attempted")

    monkeypatch.setattr(wave.Wave_read, "readframes", forbidden_read)
    with pytest.raises(ValueError, match="PCM input"):
        validate_wav(speech)


def test_audio_callback_runs_before_later_sentence_is_synthesized(speech, tmp_path):
    events = []

    async def transcribe(*args):
        return "hello"

    async def respond(*args):
        return "First. Second."

    async def synthesize(text, language):
        events.append(("synthesize", text))
        return pcm()

    async def received(path):
        assert validate_wav(path) > 0
        events.append(("audio", path.name))

    asyncio.run(
        VoicePipeline(transcribe, respond, synthesize).run(
            speech, "en", tmp_path, on_audio=received
        )
    )
    assert [kind for kind, _ in events] == ["synthesize", "audio", "synthesize", "audio"]


def test_no_project_intent_gets_a_different_real_agent_response():
    async def scenario():
        respond = scripted_responder()
        greeting = await respond("Hello", "en")
        project = await respond("Help with a project", "en")
        assert greeting != project
        assert "How can I help" in greeting

    asyncio.run(scenario())
