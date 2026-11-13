"""Replay signed fictional Voice webhooks locally; no calls, messages or server."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import urlencode

from qagents import Agent, Limits, ToolCallPart, tool
from qagents.models.testing import FunctionModel

from qagent_gym.integrations.twilio import InboundCall, TwilioVoiceWebhook

FIXTURE_URL = "https://voice.example.invalid/inbound?tenant=fixture"
FIXTURE_TOKEN = "public-test-token-never-a-real-credential"
FIXTURE_ACCOUNT = "AC" + "0" * 32


def support_responder():
    calls = []

    @tool(read_only=True)
    async def opening_hours() -> str:
        """Read fictional public opening hours; no caller authentication required."""
        calls.append("opening_hours")
        return "Our fictional desk opens Monday to Friday, nine to five."

    def policy(request):
        if request.messages[-1].role == "tool":
            part = request.messages[-1].tool_results[0]
            if part.is_error:
                raise ValueError("support tool failed")
            return part.content
        return ToolCallPart(id="hours", name="opening_hours", arguments={})

    agent = Agent(FunctionModel(policy), name="inbound-support", tools=[opening_hours])

    async def respond(event: InboundCall) -> str:
        # Phone metadata is deliberately not an application identity or tool credential.
        result = await agent.run(
            json.dumps({"question": event.speech or "opening hours", "digits": event.digits}),
            limits=Limits(max_turns=3, timeout_s=5, max_total_tokens=1500),
        )
        if result.status != "completed" or not isinstance(result.output, str):
            raise RuntimeError("support agent did not complete")
        return result.output

    return respond, calls


async def run_demo() -> dict:
    from twilio.request_validator import RequestValidator

    respond, calls = support_responder()
    webhook = TwilioVoiceWebhook(
        public_url=FIXTURE_URL,
        auth_token=FIXTURE_TOKEN,
        account_sid=FIXTURE_ACCOUNT,
        respond=respond,
    )
    signer = RequestValidator(FIXTURE_TOKEN)
    fixture = {
        "AccountSid": FIXTURE_ACCOUNT,
        "CallSid": "CA" + "1" * 32,
        "From": "+12025550101",
        "To": "+12025550102",
        "Direction": "inbound",
        "CallStatus": "ringing",
    }
    replies = []
    for params in (
        fixture,
        {**fixture, "CallStatus": "in-progress", "SpeechResult": "Opening hours?"},
    ):
        response = await webhook.handle(
            method="POST",
            content_type="application/x-www-form-urlencoded",
            body=urlencode(params).encode(),
            signature=signer.compute_signature(FIXTURE_URL, params),
        )
        replies.append({"status": response.status_code, "twiml": response.body})
    before = len(calls)
    refused = await webhook.handle(
        method="POST",
        content_type="application/x-www-form-urlencoded",
        body=urlencode(fixture).encode(),
        signature="invalid",
    )
    return {
        "mode": "offline signed fixture replay; no real phone call or public endpoint",
        "responses": replies,
        "tool_executions": calls,
        "invalid_signature_status": refused.status_code,
        "invalid_signature_invoked_agent": len(calls) != before,
        "caller_authenticated": False,
    }


def main() -> int:
    try:
        report = asyncio.run(run_demo())
    except ImportError:
        raise SystemExit("Install the optional twilio SDK to run signed local replay.") from None
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
