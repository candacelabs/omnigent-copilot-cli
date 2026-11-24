"""Mirror the pinned Copilot TUI session's JSONL transcript into Omnigent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NotRequired, TypedDict, cast
from uuid import NAMESPACE_URL, uuid5

import httpx

from omnigent.inner.native_attachments import ATTACHMENT_MARKER_STRIP_PATTERN
from omnigent.native._native_post_delivery import post_external_session_status
from omnigent.runtime.tool_output import cap_tool_output
from omnigent.util.json_types import JsonObject

_logger = logging.getLogger(__name__)
_STATE_FILE = "copilot-forwarder.sqlite3"
_ATTACHMENT = re.compile(ATTACHMENT_MARKER_STRIP_PATTERN)
_Event = tuple[str, Mapping[str, object]]


class _Status(TypedDict):
    status: str
    response_id: str | None
    output: NotRequired[str]
    turn_completed: NotRequired[bool]


@dataclass
class _Turn:
    response_id: str | None = None
    awaiting_tools: bool = False
    output: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _Cursor:
    offset: int = 0
    anchor: str = ""


def _key(value: str) -> str:
    return f"copilot:{uuid5(NAMESPACE_URL, value)}"


def _item(
    item_type: str, data: Mapping[str, object], response_id: str | None, source_id: str
) -> _Event:
    return "external_conversation_item", {
        "item_type": item_type,
        "item_data": data,
        "response_id": response_id,
        "source_id": source_id,
    }


def _finish(turn: _Turn, status: str, completed: bool, output: str = "") -> _Event:
    data = {
        "response_id": turn.response_id,
        "status": status,
        "output": output or turn.output,
        "turn_completed": completed,
    }
    turn.response_id = None
    turn.output = ""
    turn.awaiting_tools = False
    return "external_session_status", data


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _tokens(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _usage(record: JsonObject, turn: _Turn, *, parent: bool) -> JsonObject:
    data: JsonObject = {}
    if (count := _tokens(record.get("inputTokens"))) is not None:
        turn.input_tokens += count
        data["cumulative_input_tokens"] = turn.input_tokens
        if parent:
            data["context_tokens"] = count
    if (count := _tokens(record.get("outputTokens"))) is not None:
        turn.output_tokens += count
        data["cumulative_output_tokens"] = turn.output_tokens
    if parent and (window := _tokens(record.get("maxPromptTokens"))):
        data["context_window"] = window
    # Copilot's cost and nano-AIU fields measure Copilot billing units, not USD.
    return data


def _project(event: JsonObject, turn: _Turn, agent_name: str, copilot_id: str) -> list[_Event]:
    """Translate a persisted vendor event. The caller commits the new turn state after delivery."""
    kind = event.get("type")
    data = event.get("data") or {}
    if not isinstance(data, dict):
        return []
    parent = not event.get("agentId")
    if kind == "session.usage_record" and isinstance(data.get("usage"), dict):
        usage = _usage(data["usage"], turn, parent=parent)
        return [("external_session_usage", usage)] if usage else []
    # Subagents share the same physical JSONL. Their private prompts, calls and
    # step boundaries belong to the parent's task tool, not to this conversation.
    if not parent:
        return []
    source = _key(f"{copilot_id}/{event['id']}")
    if kind == "user.message":
        original = _text(data.get("content"))
        if not original.strip():
            return []
        text = _ATTACHMENT.sub("", original).strip()
        blocks = [{"type": "input_text", "text": text}] if text else []
        # Even an attachment-only prompt must settle the queued web upload.
        interaction = data.get("interactionId")
        response = _key(f"{copilot_id}/{interaction}") if interaction else source
        return [_item("message", {"role": "user", "content": blocks}, response, source)]
    if kind == "assistant.turn_start":
        interaction = data.get("interactionId")
        response = (
            _key(f"{copilot_id}/{interaction}") if interaction else turn.response_id or source
        )
        if response == turn.response_id:
            return []
        events: list[_Event] = [_finish(turn, "idle", False)] if turn.response_id else []
        turn.response_id = response
        events.append(("external_session_status", {"status": "running", "response_id": response}))
        return events
    if kind == "assistant.message":
        if not turn.response_id:
            turn.response_id = source
        turn.awaiting_tools = bool(data.get("toolRequests"))
        text = _text(data.get("content"))
        if not text.strip():
            return []
        turn.output = cap_tool_output(text)
        item = {
            "role": "assistant",
            "agent": agent_name,
            "content": [{"type": "output_text", "text": text}],
        }
        return [_item("message", item, turn.response_id, source)]
    if kind in {"tool.execution_start", "tool.execution_complete"}:
        call_id = data.get("toolCallId")
        if not isinstance(call_id, str) or not call_id:
            return []
        if kind == "tool.execution_start":
            turn.awaiting_tools = True
            tool = {
                "agent": agent_name,
                "name": data.get("toolName") or data.get("toolTitle") or "tool",
                "arguments": json.dumps(data.get("arguments") or {}, ensure_ascii=False),
                "call_id": call_id,
            }
            return [_item("function_call", tool, turn.response_id, source)]
        result = data.get("error", "") if data.get("success") is False else data.get("result", "")
        if isinstance(result, dict):
            # Copilot retains provider metadata and even the original MCP structuredContent
            # beside the visible result when a postToolUse hook redacts it.
            result = result.get("content", result.get("message", ""))
        output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return [
            _item(
                "function_call_output",
                {"call_id": call_id, "output": cap_tool_output(output)},
                turn.response_id,
                source,
            )
        ]
    if kind == "session.error":
        message = str(data.get("message") or data.get("error") or "Copilot CLI reported an error")
        err = {
            "source": "harness",
            "code": str(data.get("errorType") or "copilot_error"),
            "message": message,
        }
        events = [_item("error", err, turn.response_id, source)]
        if not data.get("isRecoverable"):
            events.append(_finish(turn, "failed", False, message))
        return events
    if kind == "assistant.turn_end" and turn.response_id and not turn.awaiting_tools:
        # Copilot emits turn_end after each model/tool step within an interaction.
        return [_finish(turn, "idle", True)]
    if kind in {"abort", "session.shutdown"} and turn.response_id:
        return [_finish(turn, "idle", False)]
    return []


def _read_batch(path: Path, cursor: _Cursor) -> list[tuple[_Cursor, JsonObject]]:
    """Read complete records, detecting a replaced or truncated transcript at the cursor."""
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return []
    with stream:
        offset = cursor.offset
        if offset:
            stream.seek(max(0, offset - 64))
            previous = stream.read(min(offset, 64))
            if hashlib.sha256(previous).hexdigest() != cursor.anchor:
                offset = 0
        stream.seek(offset)
        records = []
        for _ in range(64):
            raw = stream.readline()
            if not raw.endswith(b"\n"):
                break
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError):
                _logger.warning("Skipping malformed Copilot transcript record at byte %d", offset)
                value = {}
            offset = stream.tell()
            # A short record needs trailing bytes from its predecessor.
            if len(raw) >= 64:
                ending = raw[-64:]
            else:
                stream.seek(max(0, offset - 64))
                ending = stream.read(min(offset, 64))
            anchor = hashlib.sha256(ending).hexdigest()
            records.append((_Cursor(offset, anchor), value if isinstance(value, dict) else {}))
        return records


def clear_copilot_bridge_state(bridge_dir: Path) -> None:
    """Forget the checkpoint when deleting a terminal's bridge."""
    (bridge_dir / _STATE_FILE).unlink(missing_ok=True)


async def _post(client: httpx.AsyncClient, session_id: str, event: _Event) -> None:
    kind, data = event
    if kind == "external_session_status":
        await post_external_session_status(client, session_id=session_id, **cast(_Status, data))
        return
    response = await client.post(
        f"/v1/sessions/{session_id}/events", json={"type": kind, "data": data}
    )
    response.raise_for_status()


async def forward_copilot_events_to_session(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    bridge_dir: Path,
    agent_name: str,
    copilot_session_id: str,
    events_file: Path | None = None,
    poll_interval_s: float = 0.35,
    auth: httpx.Auth | None = None,
) -> None:
    """Tail one pinned transcript; persist offset, active turn and delivered IDs together."""
    from omnigent.harnesses.copilot_native.bridge import session_events_path

    path = events_file or session_events_path(copilot_session_id)
    bridge_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with closing(sqlite3.connect(bridge_dir / _STATE_FILE)) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS checkpoint"
            "(session TEXT PRIMARY KEY, cursor TEXT, turn TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS delivered"
            "(session TEXT, id TEXT, PRIMARY KEY(session, id)) WITHOUT ROWID"
        )
        saved = db.execute(
            "SELECT cursor, turn FROM checkpoint WHERE session=?", (copilot_session_id,)
        ).fetchone()
        cursor, turn = (
            (_Cursor(**json.loads(saved[0])), _Turn(**json.loads(saved[1])))
            if saved
            else (_Cursor(), _Turn())
        )
        async with httpx.AsyncClient(
            base_url=base_url, headers=headers, auth=auth, timeout=30
        ) as client:
            while True:
                batch = await asyncio.to_thread(_read_batch, path, cursor)
                for next_cursor, event in batch:
                    event_id = event.get("id")
                    next_turn = replace(turn)
                    if (
                        isinstance(event_id, str)
                        and event_id
                        and not db.execute(
                            "SELECT 1 FROM delivered WHERE session=? AND id=?",
                            (copilot_session_id, event_id),
                        ).fetchone()
                    ):
                        for outgoing in _project(event, next_turn, agent_name, copilot_session_id):
                            await _post(client, session_id, outgoing)
                    with db:
                        if isinstance(event_id, str) and event_id:
                            db.execute(
                                "INSERT OR IGNORE INTO delivered VALUES(?,?)",
                                (copilot_session_id, event_id),
                            )
                        db.execute(
                            "INSERT OR REPLACE INTO checkpoint VALUES(?,?,?)",
                            (
                                copilot_session_id,
                                json.dumps(asdict(next_cursor)),
                                json.dumps(asdict(next_turn)),
                            ),
                        )
                    cursor, turn = next_cursor, next_turn
                if len(batch) < 64:
                    await asyncio.sleep(poll_interval_s)


async def supervise_copilot_forwarder(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    bridge_dir: Path,
    agent_name: str,
    copilot_session_id: str,
    events_file: Path | None = None,
    poll_interval_s: float = 0.35,
    auth: httpx.Auth | None = None,
) -> None:
    """Restart at the last acknowledged event after a transient delivery or file error."""
    delay = 1.0
    loop = asyncio.get_running_loop()
    while True:
        started = loop.time()
        try:
            await forward_copilot_events_to_session(
                base_url=base_url,
                headers=headers,
                session_id=session_id,
                bridge_dir=bridge_dir,
                agent_name=agent_name,
                copilot_session_id=copilot_session_id,
                events_file=events_file,
                poll_interval_s=poll_interval_s,
                auth=auth,
            )
        except (OSError, sqlite3.Error, httpx.HTTPError):
            _logger.warning(
                "Copilot transcript forwarding interrupted for %s", session_id, exc_info=True
            )
        if loop.time() - started > 60:
            delay = 1.0
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30)
