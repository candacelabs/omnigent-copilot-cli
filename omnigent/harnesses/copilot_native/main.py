"""Create or reattach a runner-owned GitHub Copilot terminal from the Omnigent CLI."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import httpx
import yaml

from omnigent._runner_startup import runner_startup_progress
from omnigent._wrapper_labels import COPILOT_NATIVE_WRAPPER_VALUE, WRAPPER_LABEL_KEY
from omnigent.conversation_browser import conversation_url, open_conversation_link_if_enabled
from omnigent.entities.session_resources import terminal_resource_id
from omnigent.host.daemon_launch import (
    error_text,
    launch_or_reuse_daemon_runner,
    open_daemon_client,
    wait_for_host_online,
    wait_for_runner_online,
)
from omnigent.host.session_env import copilot_native_env
from omnigent.native._native_resume_hint import echo_native_resume_hint
from omnigent.native.native_coding_agents import native_shell_terminal_spec
from omnigent.native.native_terminal import (
    DAEMON_HOST_ONLINE_TIMEOUT_S,
    DAEMON_RUNNER_ONLINE_TIMEOUT_S,
    bind_session_runner,
    post_initial_prompt,
    request_with_429_retry,
    terminal_attach_url,
    url_component,
)

_AGENT = "copilot-native-ui"
_TERMINAL_ID = terminal_resource_id("copilot", "main")
_SESSION_LABELS = {"omnigent.ui": "terminal", WRAPPER_LABEL_KEY: COPILOT_NATIVE_WRAPPER_VALUE}


def validate_copilot_args(args: Sequence[str]) -> None:
    """Keep native conversation, working-directory and interactive-mode ownership in Omnigent."""
    reserved = {
        "--acp",
        "--connect",
        "--continue",
        "--interactive",
        "--prompt",
        "--resume",
        "--session-id",
        "--stdio",
        "--worktree",
        "--",
        "-C",
        "-i",
        "-p",
        "-r",
    }
    for arg in args:
        option = arg[:2] if arg[:2] in {"-C", "-i", "-p", "-r"} else arg.split("=", 1)[0]
        if option in reserved:
            raise ValueError(
                f"Copilot option {option!r} conflicts with the Omnigent terminal session."
            )


def _materialize_copilot_agent_spec(tmpdir: Path) -> Path:
    """Write the portable agent bundle used by the CLI and the web agent picker."""
    path = tmpdir / f"{_AGENT}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": _AGENT,
                "prompt": "GitHub Copilot is running in this session's terminal.",
                "executor": {"harness": "copilot-native"},
                "spawn": True,
                "os_env": {"type": "caller_process", "cwd": ".", "sandbox": {"type": "none"}},
                "terminals": native_shell_terminal_spec(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _response_body(response: httpx.Response, action: str) -> dict[str, object]:
    if response.is_error:
        raise click.ClickException(
            f"{action} failed (HTTP {response.status_code}): {error_text(response)}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise click.ClickException(f"{action} returned an invalid response.")
    return payload


async def _ensure_terminal(client: httpx.AsyncClient, session_id: str) -> dict[str, object]:
    # The runner checks the private ownership marker even when a generic shell
    # already exists with the same public terminal name and session key.
    return _response_body(
        await client.post(
            f"/v1/sessions/{url_component(session_id)}/resources/terminals",
            json={"terminal": "copilot", "session_key": "main", "ensure_native_terminal": True},
        ),
        "Starting GitHub Copilot terminal",
    )


async def _prepare_terminal(
    client: httpx.AsyncClient,
    *,
    host_id: str,
    session_id: str | None,
    bundle: bytes | None,
    model: str | None,
    extra_args: tuple[str, ...],
    workspace: str,
) -> tuple[str, dict[str, object]]:
    """Create/bind a new session or return the live main terminal of an existing one."""
    launch_options: dict[str, object] = {}
    if model:
        launch_options["model_override"] = model
    if extra_args:
        launch_options["terminal_launch_args"] = list(extra_args)
    fresh = session_id is None
    if session_id is None:
        if bundle is None:
            raise ValueError("A new GitHub Copilot session requires an agent bundle.")
        metadata = {"labels": _SESSION_LABELS, "workspace": workspace, **launch_options}
        created = await request_with_429_retry(
            lambda: client.post(
                "/v1/sessions",
                data={"metadata": json.dumps(metadata)},
                files={"bundle": (f"{_AGENT}.tar.gz", bundle, "application/gzip")},
            )
        )
        new_id = _response_body(created, "Creating GitHub Copilot session").get("session_id")
        if not isinstance(new_id, str) or not new_id:
            raise click.ClickException("The server did not return a GitHub Copilot session ID.")
        session_id = new_id
    else:
        path = f"/v1/sessions/{url_component(session_id)}"
        existing = _response_body(await client.get(path), "Loading GitHub Copilot session")
        labels = existing.get("labels")
        if (
            not isinstance(labels, dict)
            or labels.get(WRAPPER_LABEL_KEY) != COPILOT_NATIVE_WRAPPER_VALUE
        ):
            raise click.ClickException(
                f"Conversation {session_id!r} is not a copilot-native session."
            )
        resource = await client.get(f"{path}/resources/terminals/{url_component(_TERMINAL_ID)}")
        unavailable = resource.status_code == 404 or (
            resource.status_code in {409, 503}
            and any(reason in error_text(resource).lower() for reason in ("not bound", "offline"))
        )
        if not unavailable:
            terminal = _response_body(resource, "Loading GitHub Copilot terminal")
            metadata = terminal.get("metadata")
            if isinstance(metadata, dict) and metadata.get("running") is True:
                if launch_options:
                    click.echo(
                        "The running GitHub Copilot terminal keeps its current launch options.",
                        err=True,
                    )
                return session_id, await _ensure_terminal(client, session_id)
        if launch_options:
            updated = await client.patch(path, json=launch_options)
            if updated.is_error:
                raise click.ClickException(
                    f"Updating GitHub Copilot session failed: {error_text(updated)}"
                )
        saved_workspace = existing.get("workspace")
        if isinstance(saved_workspace, str) and saved_workspace:
            workspace = saved_workspace
    await wait_for_host_online(client, host_id, timeout_s=DAEMON_HOST_ONLINE_TIMEOUT_S)
    runner = await launch_or_reuse_daemon_runner(
        client,
        host_id=host_id,
        session_id=session_id,
        workspace=workspace,
        fresh=fresh,
        harness_env=copilot_native_env(os.environ),
    )
    await wait_for_runner_online(client, runner, timeout_s=DAEMON_RUNNER_ONLINE_TIMEOUT_S)
    await bind_session_runner(client, session_id, runner)
    return session_id, await _ensure_terminal(client, session_id)


async def _attach_terminal(
    base_url: str, session_id: str, terminal: dict[str, object], *, headers: dict[str, str]
) -> None:
    """Attach this TTY to exactly the resource advertised to web terminal clients."""
    from omnigent.harnesses.claude_native.main import _attach_direct_tmux, attach_local_terminal

    if terminal.get("id") != _TERMINAL_ID:
        raise click.ClickException("The runner did not return the GitHub Copilot main terminal.")
    metadata = terminal.get("metadata")
    if isinstance(metadata, dict):
        socket, target = metadata.get("tmux_socket"), metadata.get("tmux_target")
        if (
            isinstance(socket, str)
            and isinstance(target, str)
            and Path(socket).is_socket()
            and shutil.which("tmux")
        ):
            await _attach_direct_tmux(Path(socket), target)
            return
    await attach_local_terminal(
        terminal_attach_url(base_url, session_id, _TERMINAL_ID), headers=headers
    )


def run_copilot_native(
    *,
    server: str | None,
    session_id: str | None,
    extra_args: tuple[str, ...] | None = None,
    model: str | None = None,
    prompt: str | None = None,
    resume_picker: bool = False,
    auto_open_conversation: bool = False,
) -> None:
    """Launch or resume the GitHub Copilot TUI through the registered Omnigent host."""
    from omnigent.chat import _bundle_agent, _remote_headers
    from omnigent.cli import _ensure_host_daemon
    from omnigent.host.identity import load_or_create_host_identity
    from omnigent.session_host import get_session_host

    if not server:
        raise click.ClickException("GitHub Copilot requires an Omnigent server.")
    base_url = server.rstrip("/")
    launch_args = extra_args or ()
    try:
        validate_copilot_args(launch_args)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    async def drive() -> None:
        selected = session_id
        headers = _remote_headers(server_url=base_url, host_id=None)
        if resume_picker and selected is None:
            from omnigent_client import OmnigentClient

            from omnigent.repl._resume_picker import pick_conversation_by_wrapper_label_from_sdk

            async with OmnigentClient(base_url=base_url, headers=headers) as client:
                selected = await pick_conversation_by_wrapper_label_from_sdk(
                    client,
                    wrapper_value=COPILOT_NATIVE_WRAPPER_VALUE,
                    agent_name=_AGENT,
                )
            if selected is None:
                return

        with runner_startup_progress(initial_message="Starting GitHub Copilot…"):
            _ensure_host_daemon(base_url)
            host = load_or_create_host_identity().host_id
            bundle = None
            if selected is None:
                with TemporaryDirectory(prefix="omnigent-copilot-") as tmp:
                    bundle = _bundle_agent(_materialize_copilot_agent_spec(Path(tmp)))
            owner = get_session_host(selected) if selected else None
            async with open_daemon_client(
                base_url, headers, owner or host, timeout=httpx.Timeout(30, read=120)
            ) as client:
                conversation, terminal = await _prepare_terminal(
                    client,
                    host_id=host,
                    session_id=selected,
                    bundle=bundle,
                    model=model,
                    extra_args=launch_args,
                    workspace=str(Path.cwd().resolve()),
                )
        headers = _remote_headers(
            server_url=base_url, host_id=get_session_host(conversation) or owner or host
        )
        click.echo(f"Web UI: {conversation_url(base_url, conversation)}", err=True)
        open_conversation_link_if_enabled(
            base_url=base_url,
            conversation_id=conversation,
            enabled=auto_open_conversation,
            warn=lambda message: click.echo(message, err=True),
        )
        if prompt:
            await post_initial_prompt(
                base_url=base_url,
                headers=headers,
                session_id=conversation,
                prompt=prompt,
                auth=None,
                harness_name="GitHub Copilot",
            )
        await _attach_terminal(base_url, conversation, terminal, headers=headers)
        echo_native_resume_hint(
            native_command="copilot-native", session_id=conversation, server=base_url
        )

    try:
        asyncio.run(drive())
    except httpx.TransportError as exc:
        raise click.ClickException(
            f"GitHub Copilot could not reach the Omnigent server at {base_url}: {exc}"
        ) from exc
