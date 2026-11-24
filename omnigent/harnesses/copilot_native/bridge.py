"""Deliver Omnigent input to its existing GitHub Copilot CLI terminal."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from filelock import FileLock

from omnigent._platform import stable_user_id
from omnigent.harnesses.claude_native.bridge import ensure_secure_dir
from omnigent.harnesses.copilot_native.hook import ACTIVE_SESSION_FILE
from omnigent.install_ledger import atomic_write_json
from omnigent.native.native_bridge_common import (
    bridge_dir_preparation_lock,
    prune_orphaned_dirs,
    write_owner_pid_marker,
)
from omnigent.util.json_types import JsonObject
from omnigent.util.uuid import normalize_uuid

if TYPE_CHECKING:
    from omnigent.inner.terminal import TerminalInstance


BRIDGE_DIR_ENV_VAR = "HARNESS_COPILOT_NATIVE_BRIDGE_DIR"
REQUEST_SESSION_ID_ENV_VAR = "HARNESS_COPILOT_NATIVE_REQUEST_SESSION_ID"
_BRIDGE_ROOT = Path(tempfile.gettempdir()) / f"omnigent-{stable_user_id()}" / "copilot-native"
_TMUX_FILE = "tmux.json"
_POLL = 0.1
_ACK_TIMEOUT = 8.0
_COMPOSER_RULE = re.compile(r"─{4,}")
_IDLE_FOOTER = re.compile(r"\b(?:Interactive|Autopilot|Plan)\s*·", re.IGNORECASE)
_BUSY_FOOTER = re.compile(r"\bctrl\+c (?:to )?interrupt\b", re.IGNORECASE)
_FOREGROUND = re.compile(r"(?:node|bun|MainThread|copilot(?:[-.][\w.-]+)?)", re.IGNORECASE)


class TmuxInfo(TypedDict):
    session_id: str
    copilot_session_id: str
    socket_path: str
    tmux_target: str
    events_path: str


class CopilotSubmissionUncertain(RuntimeError):
    """Submission started but Copilot did not confirm it. Do not resend automatically."""


def bridge_root() -> Path:
    """Return the root shared with the native MCP relay's directory validator."""
    return _BRIDGE_ROOT


def bridge_dir_for_session_id(session_id: str) -> Path:
    """Return a private bridge path for all spellings of one Omnigent conversation ID."""
    canonical = normalize_uuid(session_id) or session_id
    return bridge_root() / hashlib.sha256(canonical.encode()).hexdigest()[:32]


def copilot_session_id_for_session_id(session_id: str) -> str:
    """Derive a vendor UUID that stays stable across runner restarts and workspaces."""
    canonical = normalize_uuid(session_id) or session_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"omnigent:copilot-native:{canonical}"))


def session_events_path(copilot_session_id: str, *, copilot_home: Path | None = None) -> Path:
    """Locate the transcript for precisely the UUID supplied to Copilot's --session-id."""
    home = copilot_home or Path(os.environ.get("COPILOT_HOME") or Path.home() / ".copilot")
    return (
        home.expanduser().absolute()
        / "session-state"
        / str(uuid.UUID(copilot_session_id))
        / "events.jsonl"
    )


def prepare_bridge_dir(session_id: str) -> Path:
    """Create or reattach the bearer-protected bridge without rotating the active token."""
    bridge_dir = bridge_dir_for_session_id(session_id)
    ensure_secure_dir(bridge_dir.parent)
    with bridge_dir_preparation_lock(bridge_dir):
        ensure_secure_dir(bridge_dir)
        config = bridge_dir / "bridge.json"
        if not config.exists():
            atomic_write_json(config, {"token": secrets.token_urlsafe(32)})
        write_owner_pid_marker(bridge_dir)
    return bridge_dir


def prune_orphaned_bridge_dirs() -> int:
    """Reclaim abandoned bridges. Copilot retains its transcript in its own home."""
    return prune_orphaned_dirs(bridge_root())


def clear_active_session_for_launch(bridge_dir: Path) -> None:
    """Forget the old process's final sessionEnd only when launching a new terminal.

    Copilot defers sessionStart until the first message, including on a cold resume.
    A running pane must retain the hook's marker on every reattachment.
    """
    (bridge_dir / ACTIVE_SESSION_FILE).unlink(missing_ok=True)


def build_copilot_native_spawn_env(session_id: str) -> dict[str, str]:
    """Bind the session's HTTP harness worker to its terminal bridge."""
    return {
        BRIDGE_DIR_ENV_VAR: str(prepare_bridge_dir(session_id)),
        REQUEST_SESSION_ID_ENV_VAR: session_id,
    }


def write_tmux_target(
    bridge_dir: Path,
    *,
    session_id: str,
    socket_path: Path,
    tmux_target: str,
    copilot_session_id: str | None = None,
    events_path: Path | None = None,
    pid: int | None = None,
) -> None:
    """Publish the socket, target and transcript selected by this terminal's launcher."""
    vendor_id = copilot_session_id or copilot_session_id_for_session_id(session_id)
    payload: JsonObject = {
        "session_id": session_id,
        "copilot_session_id": vendor_id,
        "socket_path": str(socket_path),
        "tmux_target": tmux_target,
        "events_path": str((events_path or session_events_path(vendor_id)).absolute()),
    }
    if pid is not None:
        payload["pid"] = pid
    with bridge_dir_preparation_lock(bridge_dir):
        atomic_write_json(bridge_dir / _TMUX_FILE, payload)


def read_tmux_info(bridge_dir: Path) -> TmuxInfo | None:
    """Read the launcher's atomic registration, tolerating a terminal still starting."""
    try:
        data = json.loads((bridge_dir / _TMUX_FILE).read_bytes())
    except (OSError, ValueError):
        return None
    keys = ("session_id", "copilot_session_id", "socket_path", "tmux_target", "events_path")
    if not isinstance(data, dict) or not all(
        isinstance(data.get(k), str) and data[k] for k in keys
    ):
        return None
    return TmuxInfo(
        session_id=data["session_id"],
        copilot_session_id=data["copilot_session_id"],
        socket_path=data["socket_path"],
        tmux_target=data["tmux_target"],
        events_path=data["events_path"],
    )


def _wait_for_target(
    bridge_dir: Path, deadline: float, cancel: threading.Event | None = None
) -> TmuxInfo:
    while True:
        _check_cancel(cancel)
        info = read_tmux_info(bridge_dir)
        if info is not None:
            if bridge_dir != bridge_dir_for_session_id(info["session_id"]):
                raise RuntimeError("Copilot's terminal belongs to another Omnigent session")
            return info
        if time.monotonic() >= deadline:
            raise RuntimeError("Copilot's terminal has not started for this session")
        time.sleep(_POLL)


def _tmux(socket: str, *args: str, payload: bytes | None = None) -> str:
    try:
        proc = subprocess.run(
            ["tmux", "-S", socket, *args],
            input=payload,
            capture_output=True,
            timeout=3,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError("Could not reach this session's Copilot terminal") from exc
    return proc.stdout.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class _Composer:
    draft: str
    footer: str


def _composer(screen: str) -> _Composer | None:
    """Recognize the bottommost framed Copilot composer, including its mode/status footer."""
    lines = screen.rstrip().splitlines()
    if len(lines) < 4 or not _COMPOSER_RULE.fullmatch(lines[-2].strip()):
        return None
    footer = lines[-1].strip()
    if not (_IDLE_FOOTER.search(footer) or _BUSY_FOOTER.search(footer)):
        return None
    for top in range(len(lines) - 4, -1, -1):
        if _COMPOSER_RULE.fullmatch(lines[top].strip()):
            prompt, *rest = lines[top + 1 : -2]
            if not prompt.startswith("❯") or any(
                line and not line.startswith("  ") for line in rest
            ):
                return None
            return _Composer(
                "\n".join([prompt[1:].removeprefix(" "), *(line[2:] for line in rest)]), footer
            )
    return None


def native_input_ready(session_id: str, instance: TerminalInstance) -> bool:
    """Tell the terminal watcher when the live composer is empty and accepts a message."""
    screen = instance.last_pane_text() or ""
    composer = _composer(screen)
    if composer is None or composer.draft:
        return False
    info = read_tmux_info(bridge_dir_for_session_id(session_id))
    if info is None:
        return False
    try:
        _check_session(info, screen)
    except RuntimeError:
        return False
    return True


def _pane(info: TmuxInfo) -> tuple[str, _Composer | None]:
    target = info["tmux_target"]
    output = _tmux(
        info["socket_path"],
        "display-message",
        "-p",
        "-t",
        target,
        "#{pane_id}|#{pane_current_command}|#{alternate_on}|#{pane_in_mode}|#{pane_dead}",
        ";",
        "capture-pane",
        "-p",
        "-t",
        target,
    )
    meta, _, screen = output.partition("\n")
    _check_session(info, screen)
    fields = meta.split("|")
    if len(fields) != 5 or not re.fullmatch(r"%\d+", fields[0]):
        raise RuntimeError("Could not identify this session's Copilot pane")
    pane_id, command, alternate, in_mode, dead = fields
    if dead == "1":
        raise RuntimeError("Copilot has exited. Restart its terminal to continue this session.")
    active = alternate == "1" and in_mode == "0" and _FOREGROUND.fullmatch(command)
    return pane_id, _composer(screen) if active else None


def _check_session(info: TmuxInfo, screen: str = "") -> None:
    """A native /clear, /fork or /resume can change sessions without changing panes."""
    marker = bridge_dir_for_session_id(info["session_id"]) / ACTIVE_SESSION_FILE
    try:
        active = json.loads(marker.read_bytes())
    except FileNotFoundError:
        # Before the first turn Copilot defers both sessionStart and sessionEnd.
        # An early /clear nevertheless switches UUIDs and reports it on screen.
        if not re.search(r"^\s*● Started a fresh session\.\s*(?:┃)?\s*$", screen, re.MULTILINE):
            return
        active = None
    except (ValueError, OSError) as exc:
        raise RuntimeError("Could not verify the active Copilot session") from exc
    if (
        not isinstance(active, dict)
        or active.get("copilot_session_id") != info["copilot_session_id"]
    ):
        raise RuntimeError(
            "The terminal is showing a different Copilot session. "
            f"Resume {info['copilot_session_id']} in the terminal to continue here."
        )


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Copilot message delivery was cancelled")


def _wait_for_composer(
    info: TmuxInfo, deadline: float, cancel: threading.Event | None
) -> tuple[str, _Composer]:
    while True:
        _check_cancel(cancel)
        pane_id, composer = _pane(info)
        if composer is not None:
            if composer.draft:
                raise RuntimeError(
                    "Copilot has an unfinished draft. Send or clear it in the terminal first."
                )
            return pane_id, composer
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Copilot's chat input is unavailable. Complete any sign-in, trust or approval "
                "prompt in its terminal."
            )
        time.sleep(_POLL)


def _literal_content(content: str) -> str:
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise RuntimeError("Copilot requires a non-empty message")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", text):
        raise RuntimeError("Copilot chat input contains a terminal control character")
    if text.lstrip().startswith("!") or text.strip() == "$":
        raise RuntimeError("Use the Copilot terminal to enter a shell command")
    if re.match(r"/(?:clear|fork|move|new|resume|worktree)(?:\s|$)", text.lstrip(), re.I):
        raise RuntimeError(
            "This command switches Copilot sessions. Open a new Omnigent session for a separate "
            "conversation, or use Copilot's terminal controls."
        )
    return text


def _draft_matches(draft: str, text: str) -> bool:
    # Copilot renders long multiline pastes as one numbered placeholder, and
    # scrolls long wrapped lines within the composer while keeping their suffix.
    if match := re.fullmatch(r"\[Paste #\d+ - (\d+) lines\]", draft):
        return int(match[1]) == text.count("\n") + 1
    visible = "".join(draft.split())
    expected = "".join(text.split())
    return bool(visible) and (
        visible == expected or (len(visible) >= 32 and expected.endswith(visible))
    )


def _paste(
    info: TmuxInfo, pane_id: str, text: str, cancel: threading.Event | None, deadline: float
) -> None:
    socket = info["socket_path"]
    buffer = f"omnigent-copilot-{secrets.token_hex(12)}"
    _tmux(socket, "load-buffer", "-b", buffer, "-", payload=text.encode())
    try:
        # Recheck after loading the buffer, before delivering any input to the pane.
        _check_cancel(cancel)
        current_pane, editor = _pane(info)
        if current_pane != pane_id or editor is None or editor.draft:
            raise RuntimeError("Copilot's input changed before delivery; check the terminal")
        _tmux(socket, "paste-buffer", "-p", "-b", buffer, "-t", pane_id)
    finally:
        with contextlib.suppress(RuntimeError):
            _tmux(socket, "delete-buffer", "-b", buffer)
    while True:
        _check_cancel(cancel)
        current_pane, editor = _pane(info)
        if current_pane != pane_id or editor is None:
            raise RuntimeError(
                "Copilot's input changed before submission; check the terminal draft"
            )
        if _draft_matches(editor.draft, text):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Copilot did not display the expected draft. Check its terminal before retrying."
            )
        time.sleep(_POLL)


def _event_offset(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _events_since(path: Path, offset: int) -> Iterator[tuple[int, str, JsonObject]]:
    """Yield complete, top-level session events appended past a byte offset."""
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                offset = stream.tell()
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    kind = event.get("type")
                    data = event.get("data")
                    if (
                        not event.get("agentId")
                        and isinstance(kind, str)
                        and isinstance(data, dict)
                    ):
                        yield offset, kind, data
    except FileNotFoundError:
        pass


def _model_at(path: Path, offset: int = 0) -> tuple[int, str | None]:
    """The raw CLI model ID is also available when the footer shows a display name."""
    model = None
    for end, kind, data in _events_since(path, offset):
        offset = end
        if kind in {"session.start", "session.model_change"}:
            value = data.get("newModel") or data.get("selectedModel")
            if isinstance(value, str):
                model = value
    return offset, model


def _select_model(
    info: TmuxInfo,
    pane_id: str,
    editor: _Composer,
    model: str,
    deadline: float,
    cancel: threading.Event | None,
) -> None:
    if editor.footer.rsplit(maxsplit=1)[-1] == model:
        return
    events = Path(info["events_path"])
    offset, selected = _model_at(events)
    if selected == model:
        return
    _paste(info, pane_id, f"/model {model}", cancel, deadline)
    _check_cancel(cancel)
    _check_session(info)
    _tmux(info["socket_path"], "send-keys", "-t", pane_id, "Enter")
    while True:
        _check_cancel(cancel)
        current, composer = _pane(info)
        offset, changed = _model_at(events, offset)
        selected = changed or selected
        if current == pane_id and composer and not composer.draft:
            if selected == model or composer.footer.rsplit(maxsplit=1)[-1] == model:
                return
        if time.monotonic() >= deadline:
            raise RuntimeError("Copilot did not confirm the model. Check /model in its terminal.")
        time.sleep(_POLL)


def inject_user_message(
    bridge_dir: Path,
    *,
    content: str,
    timeout_s: float = 30,
    model: str | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Deliver one literal message and confirm the vendor recorded it in this session."""
    text = _literal_content(content)
    if model and not re.fullmatch(r"[a-zA-Z0-9][\w./:-]{0,199}", model, re.ASCII):
        raise RuntimeError("Invalid Copilot model identifier")
    deadline = time.monotonic() + timeout_s
    info = _wait_for_target(bridge_dir, deadline, cancel_event)
    with FileLock(str(bridge_dir / "input.lock"), timeout=timeout_s, mode=0o600):
        pane_id, editor = _wait_for_composer(info, deadline, cancel_event)
        if model:
            _select_model(info, pane_id, editor, model, deadline, cancel_event)
        events = Path(info["events_path"])
        offset = _event_offset(events)
        _paste(info, pane_id, text, cancel_event, deadline)
        _check_cancel(cancel_event)
        _check_session(info)
        try:
            _tmux(info["socket_path"], "send-keys", "-t", pane_id, "Enter")
            deadline = time.monotonic() + min(timeout_s, _ACK_TIMEOUT)
            while True:
                for end, kind, data in _events_since(events, offset):
                    offset = end
                    raw = data.get("content")
                    if (
                        kind == "user.message"
                        and isinstance(raw, str)
                        and raw.strip() == text.strip()
                    ):
                        return
                # Local slash commands can open a menu without recording a chat turn.
                if text.startswith("/"):
                    current_pane, editor = _pane(info)
                    if current_pane == pane_id and (editor is None or not editor.draft):
                        return
                if time.monotonic() >= deadline:
                    break
                time.sleep(_POLL)
        except (RuntimeError, OSError) as exc:
            raise CopilotSubmissionUncertain(
                "Copilot's submission could not be confirmed. Check the terminal before resending."
            ) from exc
        raise CopilotSubmissionUncertain(
            "Copilot has not recorded the submitted message. Check its terminal before resending."
        )


def inject_interrupt(bridge_dir: Path, *, timeout_s: float = 10) -> None:
    """Interrupt a running model turn once, leaving idle chat and modal prompts alone."""
    info = _wait_for_target(bridge_dir, time.monotonic() + timeout_s)
    pane_id, editor = _pane(info)
    if editor is None:
        raise RuntimeError("Copilot's chat is waiting for input in its terminal.")
    if _BUSY_FOOTER.search(editor.footer):
        if editor.draft:
            raise RuntimeError("Copilot has a terminal draft. Interrupt the turn in its terminal.")
        _tmux(info["socket_path"], "send-keys", "-t", pane_id, "C-c")


def kill_session(bridge_dir: Path, *, timeout_s: float = 10) -> None:
    """Stop the linked tmux session. Its Copilot history remains resumable."""
    info = _wait_for_target(bridge_dir, time.monotonic() + timeout_s)
    _tmux(info["socket_path"], "kill-session", "-t", info["tmux_target"])
