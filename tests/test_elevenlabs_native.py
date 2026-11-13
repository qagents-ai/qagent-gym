"""Optional official SDK contract: synthetic transport only, never provider inference."""

import asyncio
import io
import json
import socket
import wave

import pytest

httpx = pytest.importorskip("httpx")
sdk_module = pytest.importorskip("elevenlabs.client")

from qagent_gym.integrations.elevenlabs import ElevenLabsSynthesizer  # noqa: E402


@pytest.mark.parametrize("outcome", ["success", "oversize", "cancel", "timeout"])
def test_official_async_sdk_transport_contract_and_close(monkeypatch, outcome):
    def no_connect(*args, **kwargs):
        raise AssertionError("network connects forbidden in native SDK contract")

    async def scenario():
        # Windows creates a local socketpair while constructing its event loop.
        monkeypatch.setattr(socket.socket, "connect", no_connect)
        entered = asyncio.Event()
        seen = []

        class PCM(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                entered.set()
                if outcome in {"cancel", "timeout"}:
                    await asyncio.Event().wait()
                yield b"\x01\x00" * (24001 if outcome == "oversize" else 20)

            async def aclose(self):
                self.closed = True

        pcm = PCM()

        async def transport(request):
            seen.append(request)
            return httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=pcm)

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            client = sdk_module.AsyncElevenLabs(api_key="fixture-not-a-secret", httpx_client=http)
            synth = ElevenLabsSynthesizer(
                client,
                voice_id="voice-fixture",
                max_audio_seconds=1,
                timeout_s=0.02 if outcome == "timeout" else 2,
            )
            task = asyncio.create_task(synth("Καλημέρα", "el"))
            if outcome == "cancel":
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif outcome == "oversize":
                with pytest.raises(RuntimeError, match="duration limit"):
                    await task
            elif outcome == "timeout":
                with pytest.raises(TimeoutError):
                    await task
            else:
                with wave.open(io.BytesIO(await task), "rb") as wav:
                    assert wav.getnframes() == 20
            assert pcm.closed
        request = seen[0]
        assert request.method == "POST"
        assert request.url.path == "/v1/text-to-speech/voice-fixture/stream"
        assert request.url.params["output_format"] == "pcm_24000"
        assert json.loads(request.content)["language_code"] == "el"
        assert json.loads(request.content)["model_id"] == "eleven_flash_v2_5"
        assert len(seen) == 1

    asyncio.run(scenario())
