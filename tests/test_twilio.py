import asyncio
import socket
from urllib.parse import urlencode
from xml.etree.ElementTree import fromstring

import pytest

validator_module = pytest.importorskip("twilio.request_validator")

from qagent_gym.integrations.twilio import TwilioVoiceWebhook  # noqa: E402

URL = "https://voice.example.invalid/inbound?tenant=one%20two"
TOKEN = "public-fixture-auth-token"
ACCOUNT = "AC" + "a" * 32
PARAMS = {
    "AccountSid": ACCOUNT,
    "CallSid": "CA" + "b" * 32,
    "Direction": "inbound",
    "CallStatus": "ringing",
    "From": "+12025550101",
    "To": "+12025550102",
}


def fixture(reply="Hello & welcome <guest>.", **options):
    events = []

    async def respond(event):
        events.append(event)
        return reply

    hook = TwilioVoiceWebhook(
        public_url=URL, auth_token=TOKEN, account_sid=ACCOUNT, respond=respond, **options
    )
    return hook, events


def request(params=None, *, url=URL):
    params = PARAMS if params is None else params
    return {
        "method": "POST",
        "content_type": "application/x-www-form-urlencoded; charset=UTF-8",
        "body": urlencode(params).encode(),
        "signature": validator_module.RequestValidator(TOKEN).compute_signature(url, params),
    }


def test_official_signature_validates_before_callback_and_xml_escapes(monkeypatch):
    hook, events = fixture()

    async def scenario():
        monkeypatch.setattr(
            socket.socket, "connect", lambda *a, **k: pytest.fail("network attempted")
        )
        return await hook.handle(**request())

    response = asyncio.run(scenario())
    assert response.status_code == 200
    tree = fromstring(response.body)
    assert tree.find("Gather/Say").text == "Hello & welcome <guest>."
    assert tree.find("Gather").attrib["action"] == URL
    assert tree.find("Hangup") is not None
    assert len(events) == 1
    assert events[0].caller == PARAMS["From"]
    assert not hasattr(events[0], "principal")


def test_gather_callback_returns_say_and_hangup():
    hook, events = fixture()
    response = asyncio.run(hook.handle(**request({**PARAMS, "SpeechResult": "Opening hours?"})))
    tree = fromstring(response.body)
    assert tree.find("Say") is not None
    assert tree.find("Gather") is None
    assert events[0].speech == "Opening hours?"


@pytest.mark.parametrize(
    "override,status",
    [
        ({"signature": "wrong"}, 403),
        ({"method": "GET"}, 405),
        ({"content_type": "application/json"}, 415),
        ({"body": b"x" * 16385}, 413),
        ({"body": b"AccountSid=x&AccountSid=y"}, 400),
        ({"body": b"field=%GG"}, 400),
        ({"body": b"field=%FF"}, 400),
        ({"body": b"field=%00"}, 400),
    ],
)
def test_invalid_request_never_invokes_agent(override, status):
    hook, events = fixture()
    response = asyncio.run(hook.handle(**{**request(), **override}))
    assert response.status_code == status
    assert events == []


@pytest.mark.parametrize(
    "url",
    ["http://voice.example.invalid/inbound?tenant=one%20two", URL + "x", URL.replace("%20", "+")],
)
def test_proxy_or_encoding_url_changes_fail_signature(url):
    hook, events = fixture()
    assert asyncio.run(hook.handle(**request(url=url))).status_code == 403
    assert events == []


@pytest.mark.parametrize(
    "changes,status",
    [
        ({"AccountSid": "AC" + "f" * 32}, 403),
        ({"Direction": "outbound-api"}, 400),
        ({"CallSid": "bad"}, 400),
        ({"SpeechResult": "x" * 1001}, 400),
        ({"Digits": "hello"}, 400),
        ({"CallStatus": "completed"}, 400),
    ],
)
def test_signed_but_invalid_fields_never_invoke_agent(changes, status):
    hook, events = fixture()
    assert asyncio.run(hook.handle(**request({**PARAMS, **changes}))).status_code == status
    assert not events


def test_unknown_signed_fields_are_preserved_for_validation():
    hook, events = fixture()
    assert (
        asyncio.run(hook.handle(**request({**PARAMS, "FutureTwilioField": "value"}))).status_code
        == 200
    )
    assert len(events) == 1


@pytest.mark.parametrize("reply", ["", "x" * 601, "bad\x00xml", None])
def test_invalid_agent_reply_is_not_returned_as_twiml(reply):
    hook, _ = fixture(reply)
    response = asyncio.run(hook.handle(**request()))
    assert response.status_code == 500
    assert fromstring(response.body).tag == "Response"


def test_callback_deadline_error_redaction_and_external_cancellation():
    async def scenario():
        hook, _ = fixture(timeout_s=0.01)

        async def slow(event):
            await asyncio.Event().wait()

        hook.respond = slow
        assert (await hook.handle(**request())).status_code == 503

        async def failure(event):
            raise RuntimeError("private credential and phone details")

        hook.respond = failure
        result = await hook.handle(**request())
        assert result.status_code == 500
        assert "private" not in result.body
        hook.respond = slow
        hook.timeout_s = 10
        task = asyncio.create_task(hook.handle(**request()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_offline_example_runs_real_agent_tools_and_rejects_invalid_delivery():
    from examples.twilio_voice import run_demo

    report = asyncio.run(run_demo())
    assert [response["status"] for response in report["responses"]] == [200, 200]
    assert report["tool_executions"] == ["opening_hours", "opening_hours"]
    assert report["invalid_signature_status"] == 403
    assert report["invalid_signature_invoked_agent"] is False
    assert report["caller_authenticated"] is False
