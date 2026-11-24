"""Launch one runner-owned GitHub Copilot TUI per Omnigent conversation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from omnigent._platform import resolve_cli_binary
from omnigent.entities.session_resources import SessionResourceView, session_resource_view_to_dict
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.harness_startup_config import resolve_harness_command
from omnigent.inner.agent_env import clean_agent_env, declared_passthrough
from omnigent.inner.datamodel import OSEnvSpec, TerminalEnvSpec
from omnigent.onboarding.copilot_auth import COPILOT_HOST_ENV_VAR, COPILOT_TOKEN_ENV_VARS
from omnigent.runner.resource_registry import COPILOT_NATIVE_TERMINAL_ROLE
from omnigent.spec.types import AgentSpec

if TYPE_CHECKING:
    from omnigent.runner.native.orchestration import NativeLaunchContext

_SECRET_ENV = (
    *COPILOT_TOKEN_ENV_VARS,
    "COPILOT_PROVIDER_API_KEY",
    "COPILOT_PROVIDER_API_KEY_COMMAND",
    "COPILOT_PROVIDER_BEARER_TOKEN",
    "COPILOT_PROVIDER_HEADERS",
)


def _copilot_terminal_env(spec: AgentSpec) -> dict[str, str]:
    """Use the existing Copilot auth precedence and the runner's child env filter."""
    from omnigent.runtime.workflow import _build_copilot_spawn_env

    settings = _build_copilot_spawn_env(spec)
    env = clean_agent_env(
        allow_prefixes=("COPILOT_",),
        allow_exact=("GH_HOST",),
        deny_exact=COPILOT_TOKEN_ENV_VARS,
        extra_allowed=declared_passthrough(spec.os_env),
    )
    if token := settings.get("HARNESS_COPILOT_GITHUB_TOKEN"):
        env["COPILOT_GITHUB_TOKEN"] = token
    if host := settings.get("HARNESS_COPILOT_GITHUB_HOST"):
        env[COPILOT_HOST_ENV_VAR] = host
    return env


def _copilot_mcp_arg(bridge_dir: Path) -> str:
    """Convert the shared, session-token-authenticated stdio bridge to Copilot's schema."""
    from omnigent.harnesses.claude_native.bridge import build_mcp_config

    config = build_mcp_config(bridge_dir)
    servers = config["mcpServers"]
    assert isinstance(servers, dict)
    entry = servers["omnigent"]
    assert isinstance(entry, dict)
    entry.update(type="local", tools=["*"])
    return json.dumps(config, separators=(",", ":"))


def _copilot_transcript(
    bridge_dir: Path, vendor_id: str, workspace: Path, env: dict[str, str]
) -> Path:
    """Pin a cold-recreated process to the recorded vendor home while its history exists."""
    from omnigent.harnesses.copilot_native.bridge import read_tmux_info, session_events_path

    advertised = read_tmux_info(bridge_dir)
    if advertised and advertised["copilot_session_id"] == vendor_id:
        saved = Path(advertised["events_path"])
        if (
            saved.is_absolute()
            and saved.parts[-3:] == ("session-state", vendor_id, "events.jsonl")
            and saved.is_file()
        ):
            env["COPILOT_HOME"] = str(saved.parents[2])
            return saved
    if home := env.get("COPILOT_HOME"):
        env["COPILOT_HOME"] = str((workspace / Path(home).expanduser()).resolve())
        return session_events_path(vendor_id, copilot_home=Path(env["COPILOT_HOME"]))
    return session_events_path(vendor_id)


async def _launch_copilot(ctx: NativeLaunchContext) -> SessionResourceView:
    """Start the TUI, then advertise its actual pane and tail its matching vendor log."""
    from omnigent.harnesses.claude_native.bridge import _write_private_file
    from omnigent.harnesses.copilot_native.bridge import (
        clear_active_session_for_launch,
        copilot_session_id_for_session_id,
        prepare_bridge_dir,
        write_tmux_target,
    )
    from omnigent.harnesses.copilot_native.forwarder import supervise_copilot_forwarder
    from omnigent.harnesses.copilot_native.hook import prepare_policy_plugin
    from omnigent.harnesses.copilot_native.main import validate_copilot_args
    from omnigent.onboarding.provider_config import load_config
    from omnigent.runner._entry import _make_auth_token_factory, _RunnerDatabricksAuth
    from omnigent.runner.native.orchestration import (
        ResolvedSpec,
        _cancel_auto_forwarder_task,
        _native_startup_instructions,
        _pi_native_launch_config,
        _register_auto_forwarder_task,
        _terminal_tmux_pane,
    )

    spec = ctx.agent_spec.spec if isinstance(ctx.agent_spec, ResolvedSpec) else ctx.agent_spec
    if spec is None or ctx.ensure_comment_relay is None:
        raise RuntimeError("GitHub Copilot requires its agent spec and session tool relay.")
    command = resolve_harness_command("copilot-native", default="copilot", cfg=load_config())
    executable = resolve_cli_binary(command)
    if executable is None:
        raise OmnigentError(
            "Install GitHub Copilot CLI on the runner host or set OMNIGENT_COPILOT_PATH.",
            code=ErrorCode.HARNESS_NOT_CONFIGURED,
        )
    config = await _pi_native_launch_config(
        session_id=ctx.session_id, server_client=ctx.server_client, runtime_label="Copilot"
    )
    extras = config.terminal_launch_args or []
    try:
        validate_copilot_args(extras)
    except ValueError as exc:
        raise OmnigentError(str(exc), code=ErrorCode.HARNESS_NOT_CONFIGURED) from exc
    env = await asyncio.to_thread(_copilot_terminal_env, spec)
    vendor_id = copilot_session_id_for_session_id(ctx.session_id)
    await _cancel_auto_forwarder_task(ctx.session_id)
    bridge_dir = prepare_bridge_dir(ctx.session_id)
    events_file = _copilot_transcript(bridge_dir, vendor_id, config.workspace, env)
    await ctx.ensure_comment_relay(
        ctx.session_id, explicit_bridge_dir=bridge_dir, await_notify=False
    )
    plugin = prepare_policy_plugin(bridge_dir, session_id=ctx.session_id)
    instructions = await _native_startup_instructions(
        ctx.agent_spec, ctx.server_client, ctx.session_id, workspace=config.workspace
    )
    if instructions:
        _write_private_file(
            bridge_dir / "omnigent.instructions.md",
            "---\napplyTo: '**'\n---\n" + instructions,
        )
        existing = env.get("COPILOT_CUSTOM_INSTRUCTIONS_DIRS")
        env["COPILOT_CUSTOM_INSTRUCTIONS_DIRS"] = ",".join(
            filter(None, (existing, str(bridge_dir)))
        )
    args = [
        *extras,
        f"--session-id={vendor_id}",
        "--plugin-dir",
        str(plugin),
        "--additional-mcp-config",
        _copilot_mcp_arg(bridge_dir),
        "--secret-env-vars=" + ",".join(_SECRET_ENV),
    ]
    if model := config.model_override or spec.executor.model:
        args.append(f"--model={model}")
    if config.reasoning_effort:
        args.append(f"--reasoning-effort={config.reasoning_effort}")
    auth = _RunnerDatabricksAuth(await asyncio.to_thread(_make_auth_token_factory))
    # Both dispatch paths hold the session launch lock and return early for a live pane.
    clear_active_session_for_launch(bridge_dir)
    resource = await ctx.resource_registry.launch_required_terminal(
        session_id=ctx.session_id,
        terminal_name="copilot",
        session_key="main",
        resource_role=COPILOT_NATIVE_TERMINAL_ROLE,
        parent_os_env=spec.os_env,
        spec=TerminalEnvSpec(
            command=executable,
            args=args,
            env=env,
            inherit_env=False,
            os_env=replace(spec.os_env or OSEnvSpec(), cwd=str(config.workspace)),
            scrollback=100_000,
            tmux_allow_passthrough=True,
            tmux_start_on_attach=False,
        ),
    )
    try:
        socket, target = _terminal_tmux_pane(
            ctx.resource_registry, ctx.session_id, terminal_name="copilot", session_key="main"
        )
        if socket is None or target is None:
            raise RuntimeError("The runner did not start the GitHub Copilot terminal.")
        write_tmux_target(
            bridge_dir,
            session_id=ctx.session_id,
            socket_path=socket,
            tmux_target=target,
            copilot_session_id=vendor_id,
            events_path=events_file,
        )
        _register_auto_forwarder_task(
            ctx.session_id,
            asyncio.create_task(
                supervise_copilot_forwarder(
                    base_url=config.server_url,
                    headers={},
                    auth=auth,
                    session_id=ctx.session_id,
                    agent_name=spec.name or "copilot-native-ui",
                    bridge_dir=bridge_dir,
                    copilot_session_id=vendor_id,
                    events_file=events_file,
                ),
                name=f"copilot-forwarder-{ctx.session_id}",
            ),
        )
        ctx.publish_event(
            ctx.session_id,
            {
                "type": "session.resource.created",
                "resource": session_resource_view_to_dict(resource),
            },
        )
    except BaseException:
        await _cancel_auto_forwarder_task(ctx.session_id)
        await ctx.resource_registry.close_terminal(ctx.session_id, resource.id)
        raise
    return resource
