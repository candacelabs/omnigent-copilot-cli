"""A real Copilot CLI and the Omnigent web message endpoint share a live conversation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest

from omnigent.harnesses.copilot_native.bridge import bridge_dir_for_session_id, read_tmux_info
from tests._helpers.polling import poll_until
from tests.e2e._native_resume_helpers import (
    cli_env,
    inject_user_message,
    omnigent_console_script,
    poll_for_assistant_marker,
    spawn_cli_background,
    wait_for_conversation_id,
    wait_for_terminal_ready,
)
from tests.e2e.conftest import configure_mock_llm, get_mock_requests, set_mock_served_models


@pytest.mark.posix_only
def test_terminal_then_web_then_reattach(
    resume_test_server: str, isolated_mock_llm_server_url: str, tmp_path: Path
) -> None:
    """Check both input surfaces, persisted turns, model context and the surviving tmux PID."""
    binary = shutil.which(os.environ.get("OMNIGENT_COPILOT_PATH") or "copilot")
    if binary is None or shutil.which("tmux") is None:
        pytest.skip("The native handoff needs GitHub Copilot CLI and tmux")

    model = "mock-copilot"
    first, second = "Remember the phrase Gull Harbor.", "Which phrase did I give you?"
    reply, recalled = "Phrase saved.", "Gull Harbor."
    set_mock_served_models(isolated_mock_llm_server_url, [model])
    configure_mock_llm(
        isolated_mock_llm_server_url, [{"text": reply}, {"text": recalled}], key=model
    )
    workspace = tmp_path / "project"
    workspace.mkdir()
    env = cli_env()
    env.update(
        OMNIGENT_CONFIG_HOME=str(tmp_path / "config"),
        OMNIGENT_COPILOT_PATH=binary,
        COPILOT_HOME=str(tmp_path / "copilot"),
        COPILOT_CACHE_HOME=str(tmp_path / "cache"),
        COPILOT_OFFLINE="true",
        COPILOT_PROVIDER_TYPE="openai",
        COPILOT_PROVIDER_BASE_URL=f"{isolated_mock_llm_server_url}/v1",
        COPILOT_PROVIDER_API_KEY="loopback-fixture",
        COPILOT_MODEL=model,
        GH_CONFIG_DIR=str(tmp_path / "gh"),
    )
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "COPILOT_GITHUB_TOKEN"):
        env.pop(key, None)
    args = [str(omnigent_console_script()), "copilot-native", "--server", resume_test_server]
    with ExitStack() as cleanup, httpx.Client(base_url=resume_test_server, timeout=20) as client:
        cleanup.callback(
            subprocess.run,
            [args[0], "host", "stop", "--server", resume_test_server],
            env=env,
            cwd=workspace,
            capture_output=True,
            timeout=45,
        )
        handle = spawn_cli_background(
            [*args, "--model", model, "--no-auto-update"], env=env, cwd=str(workspace)
        )
        try:
            conversation = wait_for_conversation_id(handle, timeout=65)
            wait_for_terminal_ready(
                client, conversation_id=conversation, harness="copilot", timeout=45
            )
            bridge = bridge_dir_for_session_id(conversation)
            info = read_tmux_info(bridge)
            assert info is not None
            tmux = ["tmux", "-S", info["socket_path"]]
            target = info["tmux_target"]

            def screen() -> str:
                return subprocess.check_output(
                    [*tmux, "capture-pane", "-p", "-t", target], text=True
                )

            def pane_pid() -> str:
                return subprocess.check_output(
                    [*tmux, "display-message", "-p", "-t", target, "#{pane_pid}"], text=True
                ).strip()

            pid = pane_pid()
            # Accept Copilot's trust prompt for the test user's scratch directory.
            poll_until(
                lambda: "Do you trust" in screen() or "Interactive ·" in screen(), timeout_s=35
            )
            if "Do you trust" in screen():
                os.write(handle.fd, b"\r")
            poll_until(lambda: "Interactive ·" in screen(), timeout_s=30)
            os.write(handle.fd, first.encode())
            poll_until(lambda: first in screen(), timeout_s=15)
            os.write(handle.fd, b"\r")
            poll_for_assistant_marker(
                client, conversation_id=conversation, marker=reply, timeout=45
            )

            inject_user_message(client, conversation_id=conversation, text=second)
            poll_for_assistant_marker(
                client, conversation_id=conversation, marker=recalled, timeout=45
            )
            requests = get_mock_requests(isolated_mock_llm_server_url, key=model)
            history = [(m["role"], str(m.get("content"))) for m in requests[-1]["messages"]]
            for role, phrase in [("user", first), ("assistant", reply), ("user", second)]:
                assert any(role == author and phrase in content for author, content in history)
            items = client.get(f"/v1/sessions/{conversation}/items", params={"limit": 50})
            items.raise_for_status()
            for role, phrase in [
                ("user", first),
                ("assistant", reply),
                ("user", second),
                ("assistant", recalled),
            ]:
                assert (
                    sum(
                        row.get("role") == role and phrase in json.dumps(row.get("content"))
                        for row in items.json()["data"]
                    )
                    == 1
                )
        finally:
            handle.terminate()

        resumed = spawn_cli_background(
            [*args, "--resume", conversation], env=env, cwd=str(workspace)
        )
        try:
            assert wait_for_conversation_id(resumed, timeout=45) == conversation
            assert read_tmux_info(bridge) == info
            assert pane_pid() == pid
            assert recalled in screen()
        finally:
            resumed.terminate()
