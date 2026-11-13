# ElevenLabs and Twilio voice service adapters

These are optional, independently usable service adapters plus two local qagents
examples. Importing them creates no SDK clients and reads no credentials. Nothing
here publishes a webhook, starts a public server, places a phone call or sends a
message. Development and validation remained private and local.

The official SDK contract tests use ElevenLabs **2.78.0** and Twilio **9.13.0**.
Install the SDK only for the integration you use:

```console
python -m pip install "elevenlabs>=2.78,<3"
python -m pip install "twilio>=9.13,<10"
```

The examples run from a checkout, with qagents and qagent-gym installed; examples
are not distributed in the package wheel. There is no default paid-model call.

## ElevenLabs: streamed PCM into VoicePipeline

```console
python -m examples.elevenlabs_voice --output-dir .gym/elevenlabs --language el
```

The default needs no ElevenLabs SDK or key. A real `qagents.Agent` uses
`FunctionModel` to call the `support_greeting` tool. `VoicePipeline` then calls
`ElevenLabsSynthesizer` with a provider-shaped fixture. The fixture supplies a
canned transcript and **silence PCM**, not ASR or synthetic speech. Reports identify
this explicitly; their timing is not a service benchmark or speech-quality result.
Unique names and exclusive file creation preserve earlier output.

The adapter accepts an injected official `AsyncElevenLabs` client:

```python
import httpx
from elevenlabs.client import AsyncElevenLabs
from qagent_gym.integrations.elevenlabs import ElevenLabsSynthesizer

async def speak(configured_api_key, authorized_voice_id):
    async with httpx.AsyncClient(timeout=15) as http:
        client = AsyncElevenLabs(api_key=configured_api_key, httpx_client=http)
        synthesize = ElevenLabsSynthesizer(client, voice_id=authorized_voice_id)
        return await synthesize("Καλημέρα", "el")
```

Calling this function contacts the service. For the existing pipeline, pass
`synthesize` as its third constructor argument and keep the HTTP client alive for
the pipeline's lifetime. The adapter owns each response stream; the host owns the
SDK and HTTP client. No shared client is closed by an individual synthesis call.

Requests explicitly select `eleven_flash_v2_5`, `pcm_24000` and `en` or `el`.
Other models/languages are rejected rather than silently substituted. The
[official model list](https://elevenlabs.io/docs/overview/models) lists both
languages for Flash v2.5. The
[stream endpoint documentation](https://elevenlabs.io/docs/api-reference/text-to-speech/stream)
describes output-format selection and language enforcement; multilingual v2 does
not support that language parameter, so it is not this adapter's default.

The provider delivers raw signed 16-bit little-endian mono PCM at 24 kHz. The
adapter validates chunk types, nonempty complete samples and the configured byte
limit, then adds a proper WAV header. It does not rename MP3 bytes. Arbitrary PCM
values cannot establish whether speech is intelligible or the provider actually
honored the language; those require live listening/evaluation.

Text is limited to 600 characters and output to at most 30 seconds. A total request
deadline defaults to 15 seconds; SDK retries are disabled. Every stream is closed
on success, validation failure, provider error, timeout or cancellation. Cleanup
has an additional bounded two-second allowance. Caller cancellation propagates,
including when cleanup fails. Provider exception messages are not exposed, and
the adapter logs neither credentials nor text.

The network response is streamed into a bounded buffer, then returned as **one
complete WAV per sentence**. VoicePipeline exposes each completed sentence through
its existing `on_audio` callback. This is not token streaming, continuous playback,
Twilio Media Streams, or a measured realtime telephone integration.

An explicit live CLI exists for one short TTS request:

```console
python -m examples.elevenlabs_voice --live --voice-id YOUR_AUTHORIZED_VOICE_ID --text "Hello." --output-dir .gym/elevenlabs-live
```

It requires `ELEVENLABS_API_KEY` in the process environment and limits live text to
120 characters. It can incur provider charges and transmit that text. There is no
automatic live fallback and no voice creation or cloning. **No live ElevenLabs
synthesis was tested: credentials were not configured.**

## Twilio: signed inbound webhook and local replay

```console
python -m examples.twilio_voice
```

This needs the optional official Twilio SDK, but no credentials. It uses clearly
fictional SIDs, reserved example phone numbers, an `.invalid` URL and a public test
token to create local signed requests. The actual `RequestValidator` validates
them. The callback runs a real qagents opening-hours tool and returns TwiML. The
initial response contains `Gather`; the local speech-result callback returns
`Say` and `Hangup`. A bad signature returns 403 without invoking the agent.
There is no HTTP listener and no Twilio REST client.

`TwilioVoiceWebhook` is a framework-neutral callback seam:

```python
from qagent_gym.integrations.twilio import TwilioVoiceWebhook

def configure_hook(auth_token, account_sid, async_read_only_callback):
    return TwilioVoiceWebhook(
        public_url="https://your-configured-host.example/voice",
        auth_token=auth_token,
        account_sid=account_sid,
        respond=async_read_only_callback,
    )

async def handle_route(hook, request_method, content_type, raw_body, signature):
    result = await hook.handle(
        method=request_method,
        content_type=content_type,
        body=raw_body,
        signature=signature,
    )
    # Adapt this response to your HTTP framework without altering its content.
    return result.body, result.status_code, result.content_type
```

Bind the configured **external HTTPS URL, including its query and encoding**.
Do not reconstruct it from `Host`, `Forwarded` or `X-Forwarded-*`. A reverse proxy
must route that endpoint consistently, preserve the raw form body and signature,
and restrict access to the application appropriately. This implementation does
not configure or trust proxy headers. See Twilio's
[webhook security guidance](https://www.twilio.com/docs/usage/webhooks/webhooks-security)
and [official request validation](https://www.twilio.com/docs/usage/security).

The handler accepts bounded URL-encoded POSTs only, rejects duplicate keys and
malformed encoding, validates the signature over all submitted fields, checks the
configured account and inbound call metadata, then invokes the callback. Unknown
signed fields are retained during validation. Invalid requests never invoke the
agent. Body limit is 16 KiB, speech input 1,000 characters, reply 600 characters;
callback timeout defaults to ten seconds. TwiML text and attributes are XML escaped;
model output cannot insert verbs. This example's `Say`/`Gather` flow is English.

`InboundCall.caller` is untrusted caller metadata, **not an application principal**.
A Twilio signature establishes delivery authenticity, not caller identity or
permission to access an account. Production caller authentication, authorization,
rate limiting and durable idempotency for actions belong in the host. Signature
validation alone does not prevent replay. The provided tool is read-only and the
handler may invoke it again for a repeated signed delivery. Do not connect
state-changing tools without those controls. Callback errors return generic HTTP
errors without provider details; caller cancellation propagates.

This Twilio adapter is independent of ElevenLabs and uses Twilio `Say`, not
ElevenLabs audio. Serving private WAVs or bridging bidirectional media is not
implemented. **No live phone call, public webhook deployment or Twilio account
configuration was tested: credentials were not configured.**

## Verification boundary

```console
python -m pytest tests/test_elevenlabs.py tests/test_elevenlabs_native.py tests/test_twilio.py -q -p no:asyncio
```

The native ElevenLabs tests instantiate the installed official SDK with
`httpx.MockTransport`; synthetic streaming responses verify the actual endpoint,
payload, PCM output choice, completion, oversized data, cancellation, deadlines
and HTTP-response closure. Twilio tests use the installed official signature
validator, valid and tampered deliveries, wrong URLs/accounts, bounded malformed
forms, XML escaping and callback failures. Socket `connect` is disabled during
selected native contracts, after Windows creates the event loop's local socket
pair. This is a test guard, not a firewall or proof of all network isolation.

Without the optional SDKs, native ElevenLabs and Twilio modules are skipped by
pytest; the injected ElevenLabs tests and offline example still run. These checks
exercise local SDK contracts and qagents execution, not live provider availability,
voice quality, telephone delivery or production latency.

Validation on 13 November 2026: **52 targeted tests passed** with the isolated
SDK packages available; the base environment passed **22 tests with two optional
test modules skipped**. Ruff lint and format checks passed. Both local CLI examples
executed, including the actual qagents tool calls and invalid-signature refusal.
