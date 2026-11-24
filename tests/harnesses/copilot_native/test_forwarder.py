"""Copilot's durable transcript and native terminal status boundaries."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import httpx
import pytest

from omnigent.harnesses.copilot_native import forwarder as f
from omnigent.util.json_types import JsonObject
from tests._helpers.polling import poll_until_async


def _event(event_id: str, kind: str, **data: object) -> JsonObject:
    return {"id": event_id, "type": kind, "data": data}


def _line(event: JsonObject) -> bytes:
    return (json.dumps(event) + "\n").encode()


def test_multi_step_interaction_has_one_turn_and_correlated_cards() -> None:
    turn = f._Turn()
    records = [
        _event(
            "u",
            "user.message",
            content="Read the file.",
            transformedContent="hidden",
            interactionId="interaction-1",
        ),
        _event("s", "assistant.turn_start", interactionId="interaction-1", turnId="0"),
        _event("m", "assistant.message", content="", toolRequests=[{"name": "view"}]),
        _event(
            "t", "tool.execution_start", toolCallId="c1", toolName="view", arguments={"path": "a"}
        ),
        _event(
            "r", "tool.execution_complete", toolCallId="c1", success=True, result={"content": "ok"}
        ),
        _event("z", "assistant.turn_end", turnId="0"),
        _event("s1", "assistant.turn_start", interactionId="interaction-1", turnId="1"),
        _event("a", "assistant.message", content="It says ok.", toolRequests=[]),
        _event("z1", "assistant.turn_end", turnId="1"),
    ]
    outputs = [
        piece for record in records for piece in f._project(record, turn, "agent", "vendor")
    ]
    assert [
        payload["status"] for kind, payload in outputs if kind == "external_session_status"
    ] == ["running", "idle"]
    assert len({data["response_id"] for _, data in outputs}) == 1
    items = [data["item_data"] for kind, data in outputs if kind == "external_conversation_item"]
    assert items == [
        {"role": "user", "content": [{"type": "input_text", "text": "Read the file."}]},
        {"agent": "agent", "name": "view", "arguments": '{"path": "a"}', "call_id": "c1"},
        {"output": "ok", "call_id": "c1"},
        {
            "agent": "agent",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "It says ok."}],
        },
    ]
    assert outputs[-1][1]["turn_completed"] is True
    assert outputs[-1][1]["output"] == "It says ok."


def test_subagents_share_usage_but_keep_their_own_transcript_and_context() -> None:
    turn = f._Turn(response_id="copilot:parent", awaiting_tools=True, input_tokens=12)
    for event in (
        _event("u", "user.message", content="Instructions for the subagent."),
        _event("a", "assistant.message", content="Subagent reply", toolRequests=[]),
        _event("s", "assistant.turn_start", interactionId="nested"),
        _event("t", "tool.execution_start", toolCallId="nested-call", toolName="view"),
        _event("e", "session.error", message="Child failed."),
        _event("z", "assistant.turn_end"),
    ):
        assert not f._project({**event, "agentId": "child"}, turn, "agent", "vendor")
    child_usage = _event(
        "m",
        "session.usage_record",
        usage={
            "inputTokens": 4,
            "outputTokens": 2,
            "maxPromptTokens": 1000,
        },
    )
    assert f._project({**child_usage, "agentId": "child"}, turn, "agent", "vendor") == [
        ("external_session_usage", {"cumulative_input_tokens": 16, "cumulative_output_tokens": 2}),
    ]
    assert turn.response_id == "copilot:parent"
    assert turn.awaiting_tools


def test_attachment_only_input_reaches_the_web_upload_reconciliation() -> None:
    turn = f._Turn()
    [(_, item)] = f._project(
        _event("u", "user.message", content="[Attached: /tmp/omnigent/upload.png]"),
        turn,
        "agent",
        "vendor",
    )
    assert item["item_data"] == {"role": "user", "content": []}


@pytest.mark.parametrize("visible", ["Result withheld", ""])
def test_tool_cards_use_only_the_visible_result(visible: str) -> None:
    record = _event(
        "r",
        "tool.execution_complete",
        toolCallId="c1",
        success=True,
        result={"content": visible, "structuredContent": {"result": "original MCP output"}},
    )
    [(_, item)] = f._project(record, f._Turn(), "agent", "vendor")
    assert item["item_data"] == {"call_id": "c1", "output": visible}


def test_usage_error_and_abort() -> None:
    turn = f._Turn(response_id="copilot:turn")
    record = _event(
        "usage",
        "session.usage_record",
        usage={
            "inputTokens": 12,
            "outputTokens": 3,
            "maxPromptTokens": 128000,
            "cost": 9,
        },
    )
    f._project(record, turn, "agent", "vendor")
    assert f._project(record, turn, "agent", "vendor") == [
        (
            "external_session_usage",
            {
                "context_tokens": 12,
                "context_window": 128000,
                "cumulative_input_tokens": 24,
                "cumulative_output_tokens": 6,
            },
        )
    ]
    _, failed = f._project(
        _event("e", "session.error", errorType="rate_limit", message="Please retry."),
        turn,
        "agent",
        "vendor",
    )
    assert failed == (
        "external_session_status",
        {
            "status": "failed",
            "output": "Please retry.",
            "response_id": "copilot:turn",
            "turn_completed": False,
        },
    )
    assert not f._project(_event("end", "assistant.turn_end"), turn, "agent", "vendor")
    turn.response_id = "copilot:next"
    [aborted] = f._project(_event("ab", "abort"), turn, "agent", "vendor")
    assert aborted[1]["turn_completed"] is False


@pytest.mark.asyncio
async def test_replay_partial_line_and_repeated_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry an ambiguous post; reread a replaced file without dropping a new identical message."""
    path = tmp_path / "events.jsonl"
    first = _line(_event("one", "user.message", content="same"))
    second = _line(_event("two", "user.message", content="same"))
    path.write_bytes(first + second[:-1])
    posts = []
    accepted = asyncio.Event()

    async def post(_client: httpx.AsyncClient, session_id: str, event: f._Event) -> None:
        assert session_id == "conv_test"
        posts.append(event)
        if len(posts) == 1:
            raise httpx.ReadError("lost acknowledgement")
        accepted.set()

    monkeypatch.setattr(f, "_post", post)

    async def run() -> None:
        await f.forward_copilot_events_to_session(
            base_url="http://localhost",
            headers={},
            session_id="conv_test",
            bridge_dir=tmp_path,
            agent_name="agent",
            copilot_session_id="vendor",
            events_file=path,
            poll_interval_s=0.01,
        )

    with pytest.raises(httpx.ReadError):
        await run()
    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(accepted.wait(), timeout=3)
        assert len(posts) == 2
        assert posts[0] == posts[1]  # The API's source-derived item ID absorbs the retry.
        # Atomic replacement has a different prefix, followed by an already acknowledged record.
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b'{"type":"metadata"}\n' + first + second)
        replacement.replace(path)
        await poll_until_async(lambda: len(posts) == 3, timeout_s=3, interval_s=0.01)
        assert posts[2][1]["item_data"] == posts[0][1]["item_data"]
        assert posts[2][1]["source_id"] != posts[0][1]["source_id"]
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    f.clear_copilot_bridge_state(tmp_path)
    assert not (tmp_path / f._STATE_FILE).exists()
