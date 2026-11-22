"""Bilingual file-to-speech pipeline; providers are injected and timed separately.

The native demo uses local faster-whisper, qagents FunctionModel, and Windows TTS.
It measures WAV-ready-to-first-response-WAV, not microphone or playback latency.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import re
import shutil
import statistics
import sys
import time
import uuid
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

Transcribe = Callable[[Path, str], Awaitable[str]]
Respond = Callable[[str, str], Awaitable[str]]
Synthesize = Callable[[str, str], Awaitable[bytes]]


def validate_wav(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if not 8000 <= audio.getframerate() <= 48000 or audio.getsampwidth() not in (1, 2, 3, 4):
            raise ValueError("PCM input requires 8–48 kHz and 1–4 bytes per sample")
        duration = audio.getnframes() / audio.getframerate()
        if audio.getnchannels() not in (1, 2) or not 0 < duration <= 30:
            raise ValueError("input must contain 0–30 seconds of mono/stereo PCM speech")
        declared_bytes = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
        if declared_bytes > 12_000_000:
            raise ValueError("PCM input exceeds the 12 MB limit")
        if len(audio.readframes(audio.getnframes())) != (
            audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
        ):
            raise ValueError("truncated WAV input")
    return duration


class VoicePipeline:
    def __init__(
        self,
        transcribe: Transcribe,
        respond: Respond,
        synthesize: Synthesize,
        *,
        timeout_s: float = 15,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.transcribe, self.respond, self.synthesize = transcribe, respond, synthesize
        self.timeout_s = timeout_s

    async def run(
        self,
        audio: Path,
        language: str,
        output_dir: Path,
        *,
        on_audio: Callable[[Path], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        if language not in {"en", "el"}:
            raise ValueError("language must be en or el")
        started = time.perf_counter()
        duration = validate_wav(audio)
        paths: list[Path] = []
        try:
            async with asyncio.timeout(self.timeout_s):
                transcript = (await self.transcribe(audio, language)).strip()
                transcribed = time.perf_counter()
                if not transcript:
                    raise ValueError("speech recognition returned no text")
                reply = (await self.respond(transcript, language)).strip()
                responded = time.perf_counter()
                if not reply or len(reply) > 600:
                    raise ValueError("voice reply must contain 1–600 characters")
                # Speak complete sentences as independent WAV chunks. This is not
                # model-token streaming: the response is complete before synthesis.
                sentences = re.split(r"(?<=[.!;;])\s+", reply)
                output_dir.mkdir(parents=True, exist_ok=True)
                turn_id = uuid.uuid4().hex
                first_audio = None
                for index, sentence in enumerate(sentences):
                    payload = await self.synthesize(sentence, language)
                    path = output_dir / f"{language}-{turn_id}-{index}.wav"
                    with path.open("xb") as stream:
                        paths.append(path)
                        stream.write(payload)
                    validate_wav(path)
                    if first_audio is None:
                        first_audio = time.perf_counter()
                    if on_audio is not None:
                        await on_audio(path)
                finished = time.perf_counter()
                return {
                    "language": language,
                    "transcript": transcript,
                    "reply": reply,
                    "input_seconds": duration,
                    "audio_files": [str(p) for p in paths],
                    "asr_ms": (transcribed - started) * 1000,
                    "agent_ms": (responded - transcribed) * 1000,
                    "first_tts_ms": (first_audio - responded) * 1000,
                    "first_audio_ms": (first_audio - started) * 1000,
                    "total_ms": (finished - started) * 1000,
                }
        except BaseException:
            # These are only files created by this turn, inside the caller's directory.
            for path in paths:
                path.unlink(missing_ok=True)
            raise


class LatestTurn:
    """Cancel and await the previous turn before accepting a replacement."""

    def __init__(self, pipeline: VoicePipeline) -> None:
        self.pipeline = pipeline
        self.task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self, audio: Path, language: str, output_dir: Path) -> asyncio.Task:
        async with self._lock:
            if self.task is not None and not self.task.done():
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
            self.task = asyncio.create_task(self.pipeline.run(audio, language, output_dir))
            return self.task


class JsonWorker:
    """Persistent local JSON-lines subprocess with one in-flight request."""

    def __init__(self, *command: str, request_timeout_s: float = 15) -> None:
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("worker request timeout must be finite and positive")
        self.command = command
        self.request_timeout_s = request_timeout_s
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()

    async def __aenter__(self) -> JsonWorker:
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            limit=8 * 1024 * 1024,
        )
        try:
            async with asyncio.timeout(60):
                line = await self.process.stdout.readline()
                if not line:
                    raise RuntimeError("speech worker exited before ready; see its stderr")
                ready = json.loads(line)
            if not ready.get("ready"):
                raise RuntimeError("speech worker did not become ready")
        except BaseException:
            await self.close()
            raise
        return self

    async def request(self, payload: dict) -> dict:
        async with self.lock:
            if self.process is None or self.process.returncode is not None:
                raise RuntimeError("speech worker is closed; restart it after cancellation")
            try:
                async with asyncio.timeout(self.request_timeout_s):
                    self.process.stdin.write((json.dumps(payload) + "\n").encode())
                    await self.process.stdin.drain()
                    result = json.loads(await self.process.stdout.readline())
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    return result
            except BaseException:
                # Never reuse a stream with an abandoned response in flight.
                await self.close()
                raise

    async def close(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()

    async def __aexit__(self, *args) -> None:
        await self.close()


def scripted_responder() -> Respond:
    from qagents import Agent, Limits
    from qagents.models.testing import FunctionModel

    def policy(request):
        payload = json.loads(next(m.text for m in reversed(request.messages) if m.role == "user"))
        text = payload["transcript"].casefold()
        project = any(word in text for word in ("project", "erasmus", "πρόγραμμα", "προγραμμα"))
        if payload["language"] == "el":
            if project:
                return "Ας οργανώσουμε το πρόγραμμα μαζί. Ποιος είναι ο στόχος σου;"
            return "Γεια σου! Πώς μπορώ να βοηθήσω;"
        if project:
            return "Let us plan the project together. What is your main goal?"
        return "Hello! How can I help?"

    agent = Agent(FunctionModel(policy), name="Quentin-voice")

    async def respond(text: str, language: str) -> str:
        result = await agent.run(
            json.dumps({"transcript": text, "language": language}, ensure_ascii=False),
            limits=Limits(max_turns=2, timeout_s=5, max_total_tokens=1000),
        )
        if result.status != "completed" or not isinstance(result.output, str):
            raise ValueError(f"voice agent did not complete: {result.status}")
        return result.output

    return respond


def word_error_rate(reference: str, hypothesis: str) -> float:
    def words(text):
        return re.findall(r"\w+", text.casefold())

    reference_words, hypothesis_words = words(reference), words(hypothesis)
    row = list(range(len(hypothesis_words) + 1))
    for i, left in enumerate(reference_words, 1):
        new = [i]
        for j, right in enumerate(hypothesis_words, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = new
    return row[-1] / max(1, len(reference_words))


async def native_demo(args) -> dict:
    if sys.platform != "win32":
        raise ValueError("the bundled native synthesizer requires Windows System.Speech")
    if args.repeats < 1 or args.repeats > 20:
        raise ValueError("repeats must be 1–20")
    root = Path(__file__).parent
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        raise ValueError("PowerShell is required for Windows System.Speech")
    started = time.perf_counter()
    async with (
        JsonWorker(powershell, "-NoProfile", "-File", str(root / "voice_tts.ps1")) as tts,
        JsonWorker(
            str(args.asr_python),
            str(root / "voice_asr_worker.py"),
            "--model",
            str(args.model.resolve()),
        ) as asr,
    ):
        startup_ms = (time.perf_counter() - started) * 1000
        voices = {}

        async def synthesize(text, language):
            data = await tts.request({"text": text, "language": language})
            expected = "el-GR" if language == "el" else "en-US"
            if data.get("culture") != expected:
                raise ValueError("speech worker selected the wrong voice culture")
            voices[language] = {"name": data["voice"], "culture": data["culture"]}
            return base64.b64decode(data["wav"], validate=True)

        async def transcribe(path, language):
            data = await asr.request({"path": str(path.resolve()), "language": language})
            return data["text"]

        agent_started = time.perf_counter()
        pipeline = VoicePipeline(transcribe, scripted_responder(), synthesize)
        agent_setup_ms = (time.perf_counter() - agent_started) * 1000
        fixtures = {
            "en": "Hello, I need help with a project.",
            "el": "Γεια σου. Θέλω βοήθεια με ένα πρόγραμμα.",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        runs = []
        for language, reference in fixtures.items():
            source = args.output / f"input-{language}-{uuid.uuid4().hex}.wav"
            source.write_bytes(await synthesize(reference, language))
            for trial in range(args.repeats):
                result = await pipeline.run(source, language, args.output)
                result.update(
                    reference=reference,
                    trial=trial + 1,
                    word_error_rate=word_error_rate(reference, result["transcript"]),
                )
                runs.append(result)
        summary = {}
        for language in fixtures:
            values = sorted(r["first_audio_ms"] for r in runs if r["language"] == language)
            summary[language] = {
                "samples": len(values),
                "first_audio_median_ms": statistics.median(values),
                "first_audio_p95_ms": values[math.ceil(len(values) * 0.95) - 1],
                "all_trials_under_1500ms": max(values) <= 1500,
            }
        return {
            "mode": "local ASR and TTS; scripted qagents response",
            "speech_worker_startup_ms": startup_ms,
            "agent_setup_ms": agent_setup_ms,
            "voices": voices,
            "latency_boundary": "input WAV ready to first response WAV ready; no playback",
            "summary": summary,
            "runs": runs,
        }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-python", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, required=True, help="Downloaded local model directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    try:
        report = asyncio.run(native_demo(args))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
