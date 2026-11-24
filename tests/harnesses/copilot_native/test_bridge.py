"""Copilot's user-facing composer, private session binding and submit acknowledgement."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from omnigent.harnesses.copilot_native import bridge

_RULE = "─" * 60
_IDLE = " Interactive · Manual Approval · / commands       test-model"


def _screen(draft: str = "", footer: str = _IDLE) -> str:
    prompt = ("❯ " + draft.replace("\n", "\n  ")).rstrip()
    return "\n".join(["Current   Sessions", _RULE, prompt, _RULE, footer, ""])


@pytest.fixture
def registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(bridge, "_BRIDGE_ROOT", tmp_path / "user" / "copilot-native")
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "copilot config"))
    directory = bridge.prepare_bridge_dir("session-a")
    bridge.write_tmux_target(
        directory,
        session_id="session-a",
        socket_path=tmp_path / "tmux socket",
        tmux_target="main",
        events_path=tmp_path / "events.jsonl",
    )
    return directory


def test_private_reattach_and_distinct_session_transcripts(registered: Path) -> None:
    original = (registered / "bridge.json").read_bytes()
    token = json.loads(original)["token"]
    assert token and bridge.prepare_bridge_dir("session-a") == registered
    assert (registered / "bridge.json").read_bytes() == original
    if os.name == "posix":
        assert (registered / "bridge.json").stat().st_mode & 0o777 == 0o600
        assert registered.stat().st_mode & 0o777 == 0o700
    peer = bridge.prepare_bridge_dir("session-b")
    assert peer != registered
    assert json.loads((peer / "bridge.json").read_bytes())["token"] != token

    uuid_id = uuid.uuid4()
    aliases = [uuid_id.hex, str(uuid_id), "conv_" + uuid_id.hex]
    assert len({bridge.bridge_dir_for_session_id(alias) for alias in aliases}) == 1
    assert len({bridge.copilot_session_id_for_session_id(alias) for alias in aliases}) == 1
    env = bridge.build_copilot_native_spawn_env("session-a")
    assert env == {
        bridge.BRIDGE_DIR_ENV_VAR: str(registered),
        bridge.REQUEST_SESSION_ID_ENV_VAR: "session-a",
    }
    info = bridge.read_tmux_info(registered)
    assert info is not None
    vendor_id = info["copilot_session_id"]
    assert vendor_id == bridge.copilot_session_id_for_session_id("session-a")
    assert (
        bridge.session_events_path(vendor_id)
        == Path(os.environ["COPILOT_HOME"]) / "session-state" / vendor_id / "events.jsonl"
    )
    assert Path(info["events_path"]).name == "events.jsonl"
    assert vendor_id != bridge.copilot_session_id_for_session_id("session-b")


@pytest.mark.parametrize(
    ("screen", "state"),
    [
        ("Do you trust the files in this folder?\n❯ 1. Yes\nEnter to select", "%7|node|1|0|0"),
        ("Sign in to GitHub\nPaste the device code\n❯", "%7|copilot|1|0|0"),
        (
            "Run this shell command?\n  1. Allow once\n❯ 2. Reject\nEnter to select",
            "%7|node|1|0|0",
        ),
        (_screen("someone is typing"), "%7|node|1|0|0"),
        (_screen(), "%7|bash|1|0|0"),
        (_screen(), "%7|node|0|0|0"),
        (_screen(), "%7|node|1|1|0"),
        (_screen(), "%7|node|1|0|1"),
    ],
)
def test_unready_pane_never_receives_keys(
    registered: Path, monkeypatch: pytest.MonkeyPatch, screen: str, state: str
) -> None:
    tmux = Mock(return_value=state + "\n" + screen)
    monkeypatch.setattr(bridge, "_tmux", tmux)
    with pytest.raises(RuntimeError):
        bridge.inject_user_message(registered, content="hello", timeout_s=0)
    assert [args.args[1] for args in tmux.call_args_list] == ["display-message"]


@pytest.mark.parametrize(
    "content",
    [
        "",
        "  ",
        "\x1b[201~",
        "ctrl\x03c",
        "a\x00b",
        "\x9bA",
        "! echo hi",
        "$",
        "/resume other",
        "/new worktree",
        "/fork",
        "/clear",
    ],
)
def test_rejects_terminal_control_and_shell_modes_before_touching_tmux(
    registered: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    tmux = Mock()
    monkeypatch.setattr(bridge, "_tmux", tmux)
    with pytest.raises(RuntimeError):
        bridge.inject_user_message(registered, content=content)
    tmux.assert_not_called()


@pytest.mark.parametrize("ack", [False, True])
def test_one_literal_paste_one_submit_and_only_new_events_acknowledge(
    registered: Path, monkeypatch: pytest.MonkeyPatch, ack: bool
) -> None:
    content = 'quote "$HOME" and `symbols`\nsecond line\nthird\tcolumn'
    info = bridge.read_tmux_info(registered)
    assert info is not None
    events = Path(info["events_path"])
    row = json.dumps({"type": "user.message", "data": {"content": content}}) + "\n"
    events.write_text(row)  # An identical earlier turn is not this submission.
    draft = ""
    buffer = b""
    commands: list[tuple[str, ...]] = []

    def tmux(socket: str, *args: str, payload: bytes | None = None) -> str:
        nonlocal draft, buffer
        assert socket == info["socket_path"]
        commands.append(args)
        if args[0] == "display-message":
            return "%7|node|1|0|0\n" + _screen(draft)
        if args[0] == "load-buffer":
            assert payload == content.encode() and args[-1] == "-"
            buffer = payload
        if args[0] == "paste-buffer":
            assert "-p" in args and args[-1] == "%7"
            draft = buffer.decode()
        if args[0] == "send-keys":
            assert args[1:] == ("-t", "%7", "Enter")
            draft = ""
            with events.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "user.message",
                            "agentId": "child",
                            "data": {"content": content},
                        }
                    )
                    + "\n"
                )
                if ack:
                    handle.write(row)
        return ""

    monkeypatch.setattr(bridge, "_tmux", tmux)
    if ack:
        bridge.inject_user_message(registered, content=content.replace("\n", "\r\n"), timeout_s=0)
    else:
        with pytest.raises(bridge.CopilotSubmissionUncertain):
            bridge.inject_user_message(registered, content=content, timeout_s=0)
    names = [c[0] for c in commands]
    for operation in ("load-buffer", "paste-buffer", "send-keys", "delete-buffer"):
        assert names.count(operation) == 1


def test_modal_appearing_after_paste_is_never_confirmed(
    registered: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmux = Mock(
        side_effect=[
            "%3|copilot|1|0|0\n" + _screen(),
            "",
            "%3|copilot|1|0|0\n" + _screen(),
            "",
            "",
            "%3|copilot|1|0|0\nRun this tool?\n❯ Allow once\nEnter to select",
        ]
    )
    monkeypatch.setattr(bridge, "_tmux", tmux)
    with pytest.raises(RuntimeError, match="before submission"):
        bridge.inject_user_message(registered, content="explain", timeout_s=0)
    assert all(call.args[1] != "send-keys" for call in tmux.call_args_list)


def test_interrupt_only_a_busy_composer_and_stop_only_registered_session(
    registered: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmux = Mock(return_value="%7|node|1|0|0\n" + _screen())
    monkeypatch.setattr(bridge, "_tmux", tmux)
    bridge.inject_interrupt(registered)
    assert len(tmux.call_args_list) == 1
    busy = _screen(footer=" Retrying model request · 2s ctrl+c interrupt       test-model")
    tmux.return_value = "%7|node|1|0|0\n" + busy
    bridge.inject_interrupt(registered)
    assert tmux.call_args.args[1:] == ("send-keys", "-t", "%7", "C-c")
    bridge.kill_session(registered)
    assert tmux.call_args.args[1:] == ("kill-session", "-t", "main")


def test_cannot_send_through_registration_for_another_session(registered: Path) -> None:
    bridge.write_tmux_target(
        registered, session_id="different", socket_path=Path("/example"), tmux_target="main"
    )
    with pytest.raises(RuntimeError, match="another Omnigent session"):
        bridge.inject_user_message(registered, content="hello", timeout_s=0)


def test_switching_sessions_inside_the_same_pane_blocks_web_input(
    registered: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = registered / "copilot-active-session.json"
    terminal = Mock()
    terminal.last_pane_text.return_value = _screen()
    assert bridge.native_input_ready("session-a", terminal)
    marker.write_text(json.dumps({"copilot_session_id": str(uuid.uuid4())}))
    assert not bridge.native_input_ready("session-a", terminal)
    tmux = Mock(return_value="%3|node|1|0|0\n" + _screen())
    monkeypatch.setattr(bridge, "_tmux", tmux)
    with pytest.raises(RuntimeError, match="different Copilot session"):
        bridge.inject_user_message(registered, content="hello", timeout_s=0)
    assert [call.args[1] for call in tmux.call_args_list] == ["display-message"]
    bridge.prepare_bridge_dir("session-a")
    assert marker.exists()
    bridge.clear_active_session_for_launch(registered)
    assert not marker.exists()
    tmux.return_value = "%3|node|1|0|0\n ● Started a fresh session.    ┃\n" + _screen()
    with pytest.raises(RuntimeError, match="different Copilot session"):
        bridge.inject_user_message(registered, content="hello", timeout_s=0)
    assert all(call.args[1] == "display-message" for call in tmux.call_args_list)


def test_model_changes_use_recorded_raw_id_with_a_localized_display_name(
    registered: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = bridge.read_tmux_info(registered)
    assert info is not None
    events = Path(info["events_path"])
    events.write_text(
        json.dumps({"type": "session.start", "data": {"selectedModel": "first"}}) + "\n"
    )
    editor = bridge._Composer("", "Interactive · Manual Approval     Custom Display Name")
    paste = Mock()
    monkeypatch.setattr(bridge, "_paste", paste)
    monkeypatch.setattr(bridge, "_pane", Mock(return_value=("%3", editor)))

    def record_model(*_args: str) -> str:
        with events.open("a") as handle:
            handle.write(
                json.dumps({"type": "session.model_change", "data": {"newModel": "second"}}) + "\n"
            )
        return ""

    monkeypatch.setattr(bridge, "_tmux", record_model)
    bridge._select_model(info, "%3", editor, "first", 0, None)
    paste.assert_not_called()
    bridge._select_model(info, "%3", editor, "second", 0, None)
    assert paste.call_args.args[2] == "/model second"
