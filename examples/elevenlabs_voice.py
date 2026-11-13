"""Local ElevenLabs transport fixture with real qagents tools; live TTS is opt-in."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import uuid
import wave
from pathlib import Path

from qagents import Agent, Limits, ToolCallPart, tool
from qagents.models.testing import FunctionModel

from qagent_gym.integrations.elevenlabs import ElevenLabsSynthesizer

from .voice_agent import VoicePipeline


class FixtureSpeechClient:
    """Provider-shaped fixture producing silence, not simulated natural speech."""

    def __init__(self) -> None:
        self.text_to_speech = self
        self.requests: list[dict] = []
        self.closed = 0

    async def stream(self, **kwargs):
        self.requests.append(kwargs)
        try:
            for _ in range(3):
                await asyncio.sleep(0)
                yield b"\x00\x00" * 800
        finally:
            self.closed += 1


def responder_with_tools():
    executed = []

    @tool(read_only=True)
    async def support_greeting(language: str) -> str:
        """Read the local support desk greeting in English or Greek."""
        executed.append(language)
        return "How can we help?" if language == "en" else "Πώς μπορούμε να βοηθήσουμε;"

    def policy(request):
        if request.messages[-1].role == "tool":
            part = request.messages[-1].tool_results[0]
            if part.is_error:
                raise ValueError("greeting tool failed")
            return part.content
        language = json.loads(request.messages[-1].text)["language"]
        return ToolCallPart(
            id="greeting", name="support_greeting", arguments={"language": language}
        )

    agent = Agent(FunctionModel(policy), name="voice-support", tools=[support_greeting])

    async def respond(text: str, language: str) -> str:
        result = await agent.run(
            json.dumps({"text": text, "language": language}),
            limits=Limits(max_turns=3, timeout_s=5, max_total_tokens=1500),
        )
        if result.status != "completed" or not isinstance(result.output, str):
            raise RuntimeError("support agent did not complete")
        return result.output

    return respond, executed


async def run_demo(output_dir: Path, language: str = "en") -> dict:
    client = FixtureSpeechClient()
    synthesize = ElevenLabsSynthesizer(client, voice_id="fixture-voice")
    respond, executed = responder_with_tools()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / f"fixture-input-{uuid.uuid4().hex}.wav"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 2400)
    with source.open("xb") as handle:
        handle.write(buffer.getvalue())

    async def transcript(path, lang):
        return "Hello" if lang == "en" else "Γεια σας"

    report = await VoicePipeline(transcript, respond, synthesize).run(source, language, output_dir)
    return {
        "mode": "offline fixture: canned transcript and silence PCM, real qagents tool execution",
        "quality_or_service_latency_measured": False,
        "tool_executions": executed,
        "provider_requests": len(client.requests),
        "closed_streams": client.closed,
        "pipeline": report,
    }


async def live_speech(text: str, language: str, voice_id: str, output_dir: Path) -> dict:
    # This entry point is invoked only by --live. Never print credentials or provider errors.
    if not text.strip() or len(text) > 120:
        raise ValueError("live text must contain 1–120 characters")
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise ValueError("ELEVENLABS_API_KEY must be configured for explicit live synthesis")
    import httpx
    from elevenlabs.client import AsyncElevenLabs

    async with httpx.AsyncClient(timeout=15) as http:
        sdk = AsyncElevenLabs(api_key=key, httpx_client=http)
        payload = await ElevenLabsSynthesizer(sdk, voice_id=voice_id)(text, language)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"elevenlabs-{uuid.uuid4().hex}.wav"
    with path.open("xb") as handle:
        handle.write(payload)
    return {"mode": "live ElevenLabs TTS", "audio_file": str(path), "bytes": len(payload)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".gym/elevenlabs"))
    parser.add_argument("--language", choices=("en", "el"), default="en")
    parser.add_argument(
        "--live", action="store_true", help="explicitly send a small paid TTS request"
    )
    parser.add_argument("--voice-id")
    parser.add_argument("--text", default="Hello from the local voice example.")
    args = parser.parse_args(argv)
    if args.live and not args.voice_id:
        parser.error("--live requires an explicitly selected --voice-id")
    try:
        report = asyncio.run(
            live_speech(args.text, args.language, args.voice_id, args.output_dir)
            if args.live
            else run_demo(args.output_dir, args.language)
        )
    except (ValueError, RuntimeError, TimeoutError, ImportError, OSError):
        parser.error("voice example failed; check optional dependencies and explicit configuration")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
