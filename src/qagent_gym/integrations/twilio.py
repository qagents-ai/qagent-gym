"""Framework-neutral, signed inbound Twilio Voice webhook; no outbound API calls."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit
from xml.etree.ElementTree import Element, SubElement, tostring


@dataclass(frozen=True)
class InboundCall:
    """Signed delivery metadata, NOT an authenticated application principal.

    Caller ID can be withheld or spoofed. A valid Twilio signature does not
    authenticate the caller, authorize tools, or make requests replay-proof.
    """

    call_sid: str
    caller: str
    called: str
    speech: str = ""
    digits: str = ""


@dataclass(frozen=True)
class WebhookResponse:
    status_code: int
    body: str
    content_type: str = "application/xml; charset=utf-8"


def _xml_text(value: str) -> bool:
    return all(
        char in "\t\n\r"
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
        for char in value
    )


class TwilioVoiceWebhook:
    """Validate the exact configured external URL and form before invoking respond.

    The hosting framework must route only this endpoint here and supply raw POST
    body, Content-Type and X-Twilio-Signature. Never derive public_url from request
    Host or forwarded headers. This handles initial calls and Gather callbacks;
    it does not run a server, initiate calls or implement application login.
    """

    def __init__(
        self,
        *,
        public_url: str,
        auth_token: str,
        account_sid: str,
        respond: Callable[[InboundCall], Awaitable[str]],
        timeout_s: float = 10,
    ) -> None:
        parsed = urlsplit(public_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or any(ord(char) < 33 for char in public_url)
        ):
            raise ValueError(
                "public_url must be the exact configured HTTPS URL without credentials"
            )
        if not isinstance(auth_token, str) or not auth_token:
            raise ValueError("an explicitly configured Twilio auth token is required")
        if not isinstance(account_sid, str) or not re.fullmatch(r"AC[0-9a-fA-F]{32}", account_sid):
            raise ValueError("account_sid must be a Twilio account SID")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or not 0 < timeout_s <= 15
        ):
            raise ValueError("timeout_s must be finite, positive and at most 15")
        try:
            from twilio.request_validator import RequestValidator
        except ImportError:
            raise ImportError(
                "install the optional twilio SDK to validate Voice webhooks"
            ) from None
        self._validator = RequestValidator(auth_token)
        self.public_url, self.account_sid = public_url, account_sid
        self.respond, self.timeout_s = respond, timeout_s

    async def handle(
        self, *, method: str, content_type: str, body: bytes, signature: str
    ) -> WebhookResponse:
        """Reject invalid deliveries before invoking any agent; return escaped TwiML."""

        def refused(status: int) -> WebhookResponse:
            return WebhookResponse(status, '<?xml version="1.0" encoding="UTF-8"?><Response />')

        if method != "POST":
            return refused(405)
        if content_type.split(";", 1)[0].strip().lower() != "application/x-www-form-urlencoded":
            return refused(415)
        if not isinstance(body, bytes) or len(body) > 16384:
            return refused(413)
        if not isinstance(signature, str) or not re.fullmatch(r"[A-Za-z0-9+/]{27}=", signature):
            return refused(403)
        try:
            raw = body.decode("utf-8", errors="strict")
            if re.search(r"%(?![0-9a-fA-F]{2})", raw):
                return refused(400)
            pairs = parse_qsl(
                raw,
                keep_blank_values=True,
                strict_parsing=True,
                errors="strict",
                max_num_fields=100,
            )
            params = dict(pairs)
            if len(params) != len(pairs) or any(
                not key or len(key) > 128 or len(value) > 2048 or not _xml_text(key + value)
                for key, value in pairs
            ):
                return refused(400)
        except (ValueError, UnicodeError):
            return refused(400)
        if not self._validator.validate(self.public_url, params, signature):
            return refused(403)
        if params.get("AccountSid") != self.account_sid:
            return refused(403)
        if (
            not re.fullmatch(r"CA[0-9a-fA-F]{32}", params.get("CallSid", ""))
            or params.get("Direction") != "inbound"
            or params.get("CallStatus") not in {"ringing", "in-progress"}
            or not 1 <= len(params.get("From", "")) <= 128
            or not 1 <= len(params.get("To", "")) <= 128
            or len(params.get("SpeechResult", "")) > 1000
            or not re.fullmatch(r"[0-9*#]{0,32}", params.get("Digits", ""))
        ):
            return refused(400)
        event = InboundCall(
            params["CallSid"],
            params["From"],
            params["To"],
            params.get("SpeechResult", ""),
            params.get("Digits", ""),
        )
        try:
            async with asyncio.timeout(self.timeout_s):
                reply = await self.respond(event)
            if (
                not isinstance(reply, str)
                or not reply.strip()
                or len(reply) > 600
                or not _xml_text(reply)
            ):
                return refused(500)
        except TimeoutError:
            return refused(503)
        except Exception:
            # Do not expose provider exceptions, phone numbers or credentials in HTTP errors.
            return refused(500)
        response = Element("Response")
        if not event.speech and not event.digits:
            parent = SubElement(
                response,
                "Gather",
                input="speech dtmf",
                action=self.public_url,
                method="POST",
                speechTimeout="auto",
                timeout="5",
            )
        else:
            parent = response
        SubElement(parent, "Say", language="en-US").text = reply
        SubElement(response, "Hangup")
        return WebhookResponse(200, tostring(response, encoding="unicode", xml_declaration=True))
