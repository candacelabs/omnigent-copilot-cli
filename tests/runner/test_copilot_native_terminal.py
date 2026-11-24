"""Launch and reattach the same runner-owned GitHub Copilot TUI."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import socket
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml

from omnigent.entities.session_resources import SessionResourceView, terminal_resource_id
from omnigent.harnesses.copilot_native import bridge, forwarder, main
from omnigent.inner.datamodel import OSEnvSandboxSpec, OSEnvSpec
from omnigent.runner import _entry
from omnigent.runner.identity import RUNNER_TUNNEL_BINDING_TOKEN_ENV_VAR
from omnigent.runner.native import NativeLaunchContext, _launch_copilot, orchestration
from omnigent.runner.resource_registry import COPILOT_NATIVE_TERMINAL_ROLE, SessionResourceRegistry
from omnigent.spec.types import ApiKeyAuth
from tests.runner.native_helpers import _harness_spec


@pytest.mark.asyncio
@pytest.mark.parametrize("advertise", [True, False], ids=["shared-pane", "failed-pane"])
async def test_runner_launch_binds_vendor_session_to_one_private_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    advertise: bool,
) -> None:
    session = "conversation-example"
    monkeypatch.setattr(bridge, "_BRIDGE_ROOT", tmp_path / "native" / "copilot-native")
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "vendor"))
    monkeypatch.setenv("OMNIGENT_COPILOT_PATH", sys.executable)
    monkeypatch.setenv(RUNNER_TUNNEL_BINDING_TOKEN_ENV_VAR, "private-runner-credential")
    monkeypatch.setenv("GH_TOKEN", "ambient-token")
    monkeypatch.setattr(_entry, "_make_auth_token_factory", lambda: lambda: None)
    monkeypatch.setattr(
        orchestration,
        "_native_startup_instructions",
        AsyncMock(return_value="Follow the project conventions."),
    )
    mirror = AsyncMock()
    monkeypatch.setattr(forwarder, "supervise_copilot_forwarder", mirror)
    spec = _harness_spec("copilot-native")
    spec = dataclasses.replace(
        spec,
        executor=dataclasses.replace(
            spec.executor,
            auth=ApiKeyAuth(api_key="spec-credential"),
            model="spec-model",
        ),
        os_env=OSEnvSpec(sandbox=OSEnvSandboxSpec(type="none")),
    )
    view = SessionResourceView(
        id=terminal_resource_id("copilot", "main"),
        type="terminal",
        session_id=session,
        name="copilot",
    )
    registry = Mock(spec=SessionResourceRegistry)
    registry.terminal_registry.get.return_value = Mock(
        running=advertise,
        socket_path=tmp_path / "pane.sock",
        tmux_target="copilot:0.0",
    )
    relay = AsyncMock()
    active_marker = bridge.prepare_bridge_dir(session) / "copilot-active-session.json"
    active_marker.write_text('{"copilot_session_id":null}')

    async def launch(**kwargs: object) -> SessionResourceView:
        assert not active_marker.exists()
        relay.assert_awaited_once_with(
            session,
            explicit_bridge_dir=bridge.bridge_dir_for_session_id(session),
            await_notify=False,
        )
        return view

    registry.launch_required_terminal = AsyncMock(side_effect=launch)
    publish = Mock()
    workspace = tmp_path / "worktree"
    async with httpx.AsyncClient(
        base_url="http://localhost",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "workspace": str(workspace),
                    "terminal_launch_args": ["--no-color"],
                    "model_override": "user-model",
                },
            )
        ),
    ) as client:
        ctx = NativeLaunchContext(
            session,
            registry,
            publish,
            server_client=client,
            agent_spec=spec,
            ensure_comment_relay=relay,
        )
        if not advertise:
            with pytest.raises(RuntimeError, match="did not start"):
                await _launch_copilot(ctx)
            registry.close_terminal.assert_awaited_once_with(session, view.id)
            mirror.assert_not_called()
            publish.assert_not_called()
            return
        try:
            assert await _launch_copilot(ctx) == view
            await asyncio.sleep(0)
        finally:
            await orchestration._cancel_auto_forwarder_task(session)
    assert registry.launch_required_terminal.await_args is not None
    launch_kwargs = registry.launch_required_terminal.await_args.kwargs
    terminal = launch_kwargs["spec"]
    assert launch_kwargs["resource_role"] == COPILOT_NATIVE_TERMINAL_ROLE
    assert launch_kwargs["parent_os_env"] == spec.os_env
    assert spec.os_env is not None and terminal.os_env.sandbox is spec.os_env.sandbox
    assert terminal.os_env.cwd == str(workspace)
    assert terminal.command == sys.executable
    assert not terminal.inherit_env
    assert terminal.env["COPILOT_GITHUB_TOKEN"] == "spec-credential"
    assert (
        "GH_TOKEN" not in terminal.env and RUNNER_TUNNEL_BINDING_TOKEN_ENV_VAR not in terminal.env
    )
    assert "spec-credential" not in str(terminal.args)
    vendor_id = bridge.copilot_session_id_for_session_id(session)
    assert f"--session-id={vendor_id}" in terminal.args
    assert "--no-color" in terminal.args and "--model=user-model" in terminal.args
    bridge_dir = bridge.bridge_dir_for_session_id(session)
    assert Path(terminal.args[terminal.args.index("--plugin-dir") + 1]) == bridge_dir
    mcp = json.loads(terminal.args[terminal.args.index("--additional-mcp-config") + 1])
    relay_entry = mcp["mcpServers"]["omnigent"]
    assert relay_entry["type"] == "local"
    assert relay_entry["args"][-2:] == ["--bridge-dir", str(bridge_dir)]
    assert (
        "Follow the project conventions." in (bridge_dir / "omnigent.instructions.md").read_text()
    )
    assert terminal.env["COPILOT_CUSTOM_INSTRUCTIONS_DIRS"].split(",")[-1] == str(bridge_dir)
    info = bridge.read_tmux_info(bridge_dir)
    assert (
        info and info["tmux_target"] == "copilot:0.0" and info["copilot_session_id"] == vendor_id
    )
    assert (
        Path(info["events_path"])
        == tmp_path / "vendor" / "session-state" / vendor_id / "events.jsonl"
    )
    assert mirror.await_args is not None
    assert mirror.await_args.kwargs["events_file"] == Path(info["events_path"])
    assert mirror.await_args.kwargs["copilot_session_id"] == vendor_id
    assert publish.call_args.args[1]["resource"]["id"] == view.id


@pytest.mark.parametrize("flag", ["--session-id=other", "--resume", "-C/other", "-p", "--acp"])
def test_incompatible_vendor_session_controls_are_rejected(flag: str) -> None:
    with pytest.raises(ValueError, match="Omnigent terminal session"):
        main.validate_copilot_args([flag])


def test_seed_agent_and_safe_interactive_options(tmp_path: Path) -> None:
    spec = yaml.safe_load(main._materialize_copilot_agent_spec(tmp_path).read_text())
    assert spec["executor"] == {"harness": "copilot-native"}
    assert spec["name"] == "copilot-native-ui" and spec["terminals"]
    main.validate_copilot_args(["--no-color", "--reasoning-effort=medium"])


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [True, False], ids=["live", "cold"])
async def test_cli_resume_preserves_the_live_resource_or_restarts_in_saved_workspace(
    monkeypatch: pytest.MonkeyPatch,
    warm: bool,
) -> None:
    calls = []
    terminal = {"id": terminal_resource_id("copilot", "main"), "metadata": {"running": True}}

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/sessions/example" and request.method == "GET":
            return httpx.Response(
                200, json={"labels": main._SESSION_LABELS, "workspace": "/saved/work"}
            )
        if request.url.path.endswith(terminal["id"]):
            return httpx.Response(200 if warm else 404, json=terminal if warm else {})
        return httpx.Response(200, json=terminal)

    launch = AsyncMock(return_value="runner")
    monkeypatch.setattr(main, "launch_or_reuse_daemon_runner", launch)
    for name in ("bind_session_runner", "wait_for_runner_online", "wait_for_host_online"):
        monkeypatch.setattr(main, name, AsyncMock())
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(respond)
    ) as client:
        assert await main._prepare_terminal(
            client,
            host_id="host",
            session_id="example",
            bundle=None,
            model="model",
            extra_args=(),
            workspace="/current/work",
        ) == ("example", terminal)
    if warm:
        launch.assert_not_awaited()
        assert ("PATCH", "/v1/sessions/example") not in calls
    else:
        assert launch.await_args is not None
        assert launch.await_args.kwargs["workspace"] == "/saved/work"
        assert ("PATCH", "/v1/sessions/example") in calls
        assert ("POST", "/v1/sessions/example/resources/terminals") in calls


@pytest.mark.asyncio
async def test_vendor_configuration_crosses_only_the_local_runner_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnigent.host import daemon_launch, session_env
    from omnigent.host.connect import _build_runner_env
    from omnigent.inner.agent_env import clean_agent_env

    monkeypatch.setattr(session_env, "_handoff_dir", lambda: tmp_path / "handoffs")
    monkeypatch.delenv("OMNIGENT_SESSION_ENV_PASSTHROUGH", raising=False)
    settings = {
        "COPILOT_HOME": str(tmp_path / "vendor"),
        "COPILOT_PROVIDER_API_KEY": "fixture-api-key",
        "COPILOT_PROVIDER_BASE_URL": "http://localhost:11555/v1",
    }
    host_values = {**settings, "HOST_ONLY_VALUE": "host-private"}
    runner_env = dict[str, str]()

    def receive(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/hosts/host/runners"
        body = json.loads(request.content)
        assert set(body) == {"workspace", "session_id", "session_env_handoff_id"}
        handed = session_env.consume_session_env_handoff(
            "conversation", body["session_env_handoff_id"]
        )
        assert handed == settings
        runner_env.update(
            _build_runner_env(
                {},
                server_url="http://localhost",
                runner_id="runner",
                binding_token="binding",
                workspace="/workspace",
                parent_pid=1,
                harness="copilot-native",
                session_env=handed,
            )
        )
        return httpx.Response(200, json={"runner_id": "runner"})

    async with httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(receive)
    ) as client:
        assert (
            await daemon_launch.launch_or_reuse_daemon_runner(
                client,
                host_id="host",
                session_id="conversation",
                workspace="/workspace",
                fresh=True,
                harness_env=session_env.copilot_native_env(host_values),
            )
            == "runner"
        )
    assert settings.items() <= runner_env.items()
    assert "COPILOT_PROVIDER_API_KEY" not in clean_agent_env(source=runner_env)
    assert not list((tmp_path / "handoffs").iterdir())

    def web_runner(harness: str) -> dict[str, str]:
        return _build_runner_env(
            host_values,
            server_url="http://localhost",
            runner_id="runner",
            binding_token="binding",
            workspace="/workspace",
            parent_pid=1,
            harness=harness,
        )

    assert settings.items() <= web_runner("copilot-native").items()
    assert not (settings.keys() & web_runner("claude-native").keys())
    assert "HOST_ONLY_VALUE" not in web_runner("copilot-native")


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [True, False], ids=["local-socket", "remote-resource"])
async def test_cli_attach_targets_same_terminal_as_the_web(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local: bool,
) -> None:
    from omnigent.harnesses.claude_native import main as attachment

    direct, websocket = AsyncMock(), AsyncMock()
    monkeypatch.setattr(attachment, "_attach_direct_tmux", direct)
    monkeypatch.setattr(attachment, "attach_local_terminal", websocket)
    monkeypatch.setattr(main.shutil, "which", lambda name: "/usr/bin/tmux")
    path = tmp_path / "copilot.sock"
    terminal: dict[str, object] = {
        "id": terminal_resource_id("copilot", "main"),
        "metadata": {"tmux_socket": str(path), "tmux_target": "main:0.0"},
    }
    with socket.socket(socket.AF_UNIX) as tmux_socket:
        if local:
            tmux_socket.bind(str(path))
        await main._attach_terminal(
            "https://example.test/omni", "conversation", terminal, headers={}
        )
    if local:
        direct.assert_awaited_once_with(path, "main:0.0")
        websocket.assert_not_called()
    else:
        direct.assert_not_called()
        websocket.assert_awaited_once_with(
            "wss://example.test/omni/v1/sessions/conversation/resources/terminals/terminal_copilot_main/attach",
            headers={},
        )


def test_cold_web_runner_reuses_the_recorded_vendor_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnigent.runner.native.copilot import _copilot_transcript

    monkeypatch.setattr(bridge, "_BRIDGE_ROOT", tmp_path / "native" / "copilot-native")
    session = "cold-conversation"
    vendor_id = bridge.copilot_session_id_for_session_id(session)
    bridge_dir = bridge.prepare_bridge_dir(session)
    env = {"COPILOT_HOME": "relative-vendor-home"}
    original = _copilot_transcript(bridge_dir, vendor_id, tmp_path, env)
    assert original == bridge.session_events_path(
        vendor_id, copilot_home=tmp_path / "relative-vendor-home"
    )
    original.parent.mkdir(parents=True)
    original.write_text('{"type":"session.start"}\n')
    bridge.write_tmux_target(
        bridge_dir,
        session_id=session,
        socket_path=tmp_path / "old.sock",
        tmux_target="old",
        copilot_session_id=vendor_id,
        events_path=original,
    )
    web_env = dict[str, str]()
    assert _copilot_transcript(bridge_dir, vendor_id, tmp_path, web_env) == original
    assert web_env["COPILOT_HOME"] == env["COPILOT_HOME"]
