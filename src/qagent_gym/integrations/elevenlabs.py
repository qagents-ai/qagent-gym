"""Bounded asynchronous ElevenLabs PCM streaming, returned as a complete WAV.

The injected client is caller-owned. Importing this module never creates an SDK
client or reads credentials. Each request stream is closed, including on cancel.
"""

from __future__ import annotations

import asyncio
import io
import math
import re
import sys
import wave
from typing import Any


class VoiceSynthesisError(RuntimeError):
    """A sanitized provider/stream failure; provider exception text is not exposed."""


class ElevenLabsSynthesizer:
    """Implement ``await synthesize(text, language) -> WAV bytes`` for VoicePipeline.

    Uses the official AsyncElevenLabs ``text_to_speech.stream`` interface. PCM is
    24 kHz, signed 16-bit little-endian mono, not MP3 renamed as WAV. Network chunks
    are bounded and assembled before returning; this is not playback streaming.
    The caller must close its SDK/HTTP client and supply a voice it may use.
    """

    def __init__(
        self,
        client: Any,
        *,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        timeout_s: float = 15,
        max_audio_seconds: float = 30,
    ) -> None:
        if not isinstance(voice_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", voice_id):
            raise ValueError("voice_id must be a bounded provider identifier")
        if model_id != "eleven_flash_v2_5":
            raise ValueError("this adapter supports eleven_flash_v2_5 with explicit en/el language")
        for name, value, maximum in (
            ("timeout_s", timeout_s, 120),
            ("max_audio_seconds", max_audio_seconds, 30),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError(f"{name} must be finite, positive and at most {maximum}")
        self._client = client
        self.voice_id, self.model_id = voice_id, model_id
        self.timeout_s = timeout_s
        self.max_pcm_bytes = int(max_audio_seconds * 24000) * 2
        if self.max_pcm_bytes < 2:
            raise ValueError("max_audio_seconds must allow at least one sample")

    async def __call__(self, text: str, language: str) -> bytes:
        if not isinstance(text, str) or not text.strip() or len(text) > 600:
            raise ValueError("speech text must contain 1–600 characters")
        if any(ord(char) < 32 and char not in "\n\t" for char in text):
            raise ValueError("speech text contains control characters")
        if language not in {"en", "el"}:
            raise ValueError("language must be en or el; no silent fallback")
        stream = None
        pcm = bytearray()
        try:
            async with asyncio.timeout(self.timeout_s):
                stream = self._client.text_to_speech.stream(
                    voice_id=self.voice_id,
                    text=text,
                    model_id=self.model_id,
                    language_code=language,
                    output_format="pcm_24000",
                    request_options={"timeout_in_seconds": self.timeout_s, "max_retries": 0},
                )
                if not hasattr(stream, "__aiter__") or not callable(
                    getattr(stream, "aclose", None)
                ):
                    raise VoiceSynthesisError("provider must supply a closable asynchronous stream")
                async for chunk in stream:
                    if not isinstance(chunk, bytes):
                        raise VoiceSynthesisError("provider returned non-byte audio")
                    if len(pcm) + len(chunk) > self.max_pcm_bytes:
                        raise VoiceSynthesisError(
                            "provider audio exceeds the configured duration limit"
                        )
                    pcm.extend(chunk)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise TimeoutError("speech synthesis exceeded its deadline") from None
        except VoiceSynthesisError:
            raise
        except Exception:
            raise VoiceSynthesisError("speech provider request failed") from None
        finally:
            pending_error = sys.exception()
            if stream is not None and callable(getattr(stream, "aclose", None)):
                try:
                    # Cleanup has its own bounded allowance after the request deadline.
                    async with asyncio.timeout(2):
                        await stream.aclose()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if pending_error is None:
                        raise VoiceSynthesisError("speech stream could not be closed") from None
        if not pcm or len(pcm) % 2:
            raise VoiceSynthesisError("provider returned empty or incomplete 16-bit PCM")
        if pcm[:4] in (b"RIFF", b"OggS") or pcm[:3] == b"ID3":
            raise VoiceSynthesisError("provider returned a container instead of requested raw PCM")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm)
        return output.getvalue()
