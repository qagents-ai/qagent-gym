import asyncio
import json
import sqlite3

import httpx
import pytest
from qagents.models import OpenAIChatModel

from examples.real_agents import Notes, endpoint_config, fixture_factory, main, run_scenario


@pytest.mark.parametrize("scenario", ["tools", "supervisor"])
def test_explicit_fixtures_execute_local_tools(tmp_path, scenario):
    result = asyncio.run(run_scenario(fixture_factory, scenario, tmp_path / "sessions.sqlite3"))
    assert result["status"] == "completed"
    assert result["local_tool_executions"] == ["quote_catalog"]
    assert "5000" in result["output"]
    assert result["stored_note_count"] == 0
    assert result["model_requests"] == (4 if scenario == "supervisor" else 2)


@pytest.mark.parametrize("approve", [True, False])
def test_approval_requires_explicit_decision_and_survives_database_reopen(tmp_path, approve):
    database = tmp_path / "sessions.sqlite3"
    paused = asyncio.run(run_scenario(fixture_factory, "approval", database))
    assert paused["status"] == "paused"
    assert paused["output"] is None
    assert paused["local_tool_executions"] == []
    assert paused["stored_note_count"] == 0
    pending = paused["pending_approvals"][0]
    assert pending["name"] == "save_note"
    result = asyncio.run(
        run_scenario(
            fixture_factory,
            "resume",
            database,
            session_id=paused["session_id"],
            decision=approve,
            tool_call_id=pending["tool_call_id"],
        )
    )
    assert result["status"] == "completed"
    assert result["stored_note_count"] == int(approve)
    assert result["local_tool_executions"] == (["save_note"] if approve else [])
    with pytest.raises(ValueError, match="pending approval"):
        asyncio.run(
            run_scenario(
                fixture_factory,
                "resume",
                database,
                session_id=paused["session_id"],
                decision=True,
                tool_call_id=pending["tool_call_id"],
            )
        )


def test_wrong_call_id_cannot_authorize_another_pending_call(tmp_path):
    database = tmp_path / "sessions.sqlite3"
    paused = asyncio.run(run_scenario(fixture_factory, "approval", database))
    with pytest.raises(ValueError, match="pending approval"):
        asyncio.run(
            run_scenario(
                fixture_factory,
                "resume",
                database,
                session_id=paused["session_id"],
                decision=True,
                tool_call_id="different-call",
            )
        )
    with sqlite3.connect(str(database) + ".notes.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM approved_notes").fetchone()[0] == 0


def test_note_receipt_replays_and_conflicting_operation_rejects(tmp_path):
    notes = Notes(tmp_path / "notes.sqlite3")
    try:
        first = notes.save("stable-operation", "Keep this note")
        assert notes.save("stable-operation", "Keep this note") == first
        with pytest.raises(ValueError, match="different note"):
            notes.save("stable-operation", "Replace it")
        assert notes.count() == 1
    finally:
        notes.close()


@pytest.mark.parametrize(
    "base,model",
    [
        (None, None),
        ("https://provider.invalid/v1", None),
        ("https://name:secret@provider.invalid/v1", "model"),
        ("http://remote.invalid/v1", "model"),
        ("https://provider.invalid/v1?api_key=secret", "model"),
    ],
)
def test_invalid_provider_configuration_never_falls_back_to_fixture(base, model):
    with pytest.raises(ValueError):
        endpoint_config(base, model)


def test_cli_without_configuration_fails_before_creating_storage(tmp_path, monkeypatch):
    monkeypatch.delenv("QAGENTS_BASE_URL", raising=False)
    monkeypatch.delenv("QAGENTS_MODEL", raising=False)
    output = tmp_path / "sessions.sqlite3"
    with pytest.raises(SystemExit) as error:
        main(["tools", "--db", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def stream_reply(*, text=None, tool_name=None, arguments=None):
    delta = {"role": "assistant"}
    finish = "stop"
    if tool_name:
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": f"call-{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }
        ]
        finish = "tool_calls"
    else:
        delta["content"] = text
    chunk = {
        "id": "fixture-response",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "configured-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    return ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()


@pytest.mark.parametrize("scenario", ["tools", "approval", "supervisor"])
def test_real_openai_chat_adapter_executes_scenarios_with_mock_http(tmp_path, scenario):
    async def run():
        requests = []

        async def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            assert request.url == "https://configured.example.invalid/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer public-fixture-token"
            assert body["model"] == "configured-model" and body["stream"] is True
            tools = {item["function"]["name"] for item in body["tools"]}
            if body["messages"][-1]["role"] == "tool":
                response = stream_reply(text="Completed using the recorded tool result.")
            elif "save_note" in tools:
                response = stream_reply(
                    tool_name="save_note", arguments={"text": "Review catalog quote."}
                )
            elif "catalog_specialist" in tools:
                response = stream_reply(
                    tool_name="catalog_specialist", arguments={"task": "Quote two lamps."}
                )
            else:
                response = stream_reply(
                    tool_name="quote_catalog", arguments={"product": "desk-lamp", "quantity": 2}
                )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=response
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model = OpenAIChatModel(
                "configured-model",
                base_url="https://configured.example.invalid/v1",
                api_key="public-fixture-token",
                http_client=client,
                max_retries=0,
            )

            def factory(role):
                return model

            database = tmp_path / "sessions.sqlite3"
            result = await run_scenario(factory, scenario, database)
            if scenario == "approval":
                assert result["status"] == "paused" and len(requests) == 1
                assert result["stored_note_count"] == 0
                result = await run_scenario(
                    factory,
                    "resume",
                    database,
                    session_id=result["session_id"],
                    decision=True,
                    tool_call_id=result["pending_approvals"][0]["tool_call_id"],
                )
                assert result["stored_note_count"] == 1
            assert result["status"] == "completed"
            assert result["local_tool_executions"] == (
                ["save_note"] if scenario == "approval" else ["quote_catalog"]
            )
            assert len(requests) == (4 if scenario == "supervisor" else 2)
            assert not client.is_closed  # Caller retains transport ownership.

    asyncio.run(run())
