"""Session-local Copilot command hooks for terminal identity and tool policies."""

from __future__ import annotations

import json
import sys
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from filelock import FileLock

from omnigent.install_ledger import atomic_write_json
from omnigent.native.native_policy_hook import (
    hook_payload_to_evaluation_request,
    is_loopback_url,
    post_evaluate_with_retry,
    read_relay_policy_config,
    relay_policy_evaluate_url,
)
from omnigent.util.json_types import JsonObject

_HOOKS = {
    "preToolUse": "PreToolUse",
    "postToolUse": "PostToolUse",
    "postToolUseFailure": "PostToolUseFailure",
}
_READ_TIMEOUT = 86_400
ACTIVE_SESSION_FILE = "copilot-active-session.json"
_LIFECYCLE_HOOKS = ("sessionStart", "sessionEnd")


def prepare_policy_plugin(bridge_dir: Path, *, session_id: str) -> Path:
    """Mount the policy hooks for this CLI invocation with its private relay directory."""
    plugin = bridge_dir.resolve()
    plugin.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_write_json(
        plugin / "plugin.json",
        {"name": "omnigent-session-policy", "version": "1.0.0", "hooks": "./hooks.json"},
    )
    hooks = {
        name: [
            {
                "type": "command",
                "exec": sys.executable,
                "args": ["-Im", __spec__.name, str(plugin), session_id, name],
                # The Omnigent ASK long poll must finish before Copilot's hook timeout.
                "timeoutSec": _READ_TIMEOUT + 120,
            }
        ]
        for name in (*_LIFECYCLE_HOOKS, *_HOOKS)
    }
    atomic_write_json(plugin / "hooks.json", {"version": 1, "hooks": hooks})
    return plugin


def _blocked(hook: str, reason: str) -> JsonObject:
    if hook == "preToolUse":
        return {"permissionDecision": "deny", "permissionDecisionReason": reason}
    if hook == "postToolUse":
        return {"modifiedResult": {"resultType": "success", "textResultForLlm": reason}}
    return {"additionalContext": reason}


def _is_relay_tool(bridge_dir: Path, name: str) -> bool:
    """Copilot exposes an Omnigent MCP entry as omnigent-<name>."""
    if not name.startswith("omnigent-"):
        return False
    try:
        manifest = json.loads((bridge_dir / "tool_relay.json").read_bytes())
        return any(
            isinstance(tool, dict) and tool.get("name") == name.removeprefix("omnigent-")
            for tool in manifest.get("tools", [])
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def evaluate(bridge_dir: Path, session_id: str, hook: str, payload: JsonObject) -> JsonObject:
    """Evaluate with the runner holding this session's credentials and policy state."""
    from omnigent.harnesses.copilot_native.bridge import copilot_session_id_for_session_id

    name = payload.get("toolName")
    if not isinstance(name, str) or not name:
        return _blocked(hook, "Copilot provided no tool name for policy evaluation.")
    expected = copilot_session_id_for_session_id(session_id)
    active = payload.get("sessionId")
    if payload.get("agentId"):
        # A subagent may identify a child session. Bind it to its foreground parent.
        try:
            active = json.loads((bridge_dir / ACTIVE_SESSION_FILE).read_bytes()).get(
                "copilot_session_id"
            )
        except (OSError, ValueError, AttributeError):
            active = None
    if active != expected:
        return _blocked(hook, "This Copilot conversation is outside the linked Omnigent session.")
    relay = read_relay_policy_config(bridge_dir)
    if relay is None or relay[2] != session_id or not is_loopback_url(relay[0]):
        return _blocked(hook, "Omnigent's session policy relay is unavailable.")
    if _is_relay_tool(bridge_dir, name):
        # The authenticated Omnigent MCP relay evaluates these phases itself.
        return {}
    args = payload.get("toolArgs", {})
    if isinstance(args, str):
        with suppress(ValueError):
            args = json.loads(args)
    native = {
        "tool_name": name,
        "tool_input": args,
        "tool_response": payload.get("toolResult"),
        "error": payload.get("error"),
    }
    request = hook_payload_to_evaluation_request(_HOOKS[hook], native)
    if request is None:
        # The shared converter recognizes Claude-style relay names. A vendor
        # tool must appear in this particular Copilot relay manifest to skip it.
        return _blocked(hook, "The tool name conflicts with Omnigent's relay namespace.")
    request["event"]["context"]["harness"] = "copilot-native"
    url, token, _ = relay
    response, _ = post_evaluate_with_retry(
        relay_policy_evaluate_url(url),
        {"Authorization": f"Bearer {token}"},
        request,
        _READ_TIMEOUT,
        "Copilot policy hook",
    )
    if response is None:
        return _blocked(hook, "Omnigent could not evaluate this tool against session policy.")
    try:
        verdict = response.json()
    except ValueError:
        verdict = None
    if not isinstance(verdict, dict):
        return _blocked(hook, "Omnigent returned an invalid tool policy response.")
    action = verdict.get("result")
    if action in ("POLICY_ACTION_ALLOW", "POLICY_ACTION_UNSPECIFIED"):
        # On ALLOW, continue through the vendor's own consent and sandbox checks.
        return {}
    return _blocked(hook, str(verdict.get("reason") or "Omnigent policy denied this tool."))


def _record_session(bridge_dir: Path, hook: str, payload: JsonObject) -> None:
    """Record the pinned TUI's foreground session, including a /clear before its next prompt."""
    if payload.get("agentId") or not isinstance(active := payload.get("sessionId"), str):
        return
    vendor_id = str(UUID(active))
    marker = bridge_dir / ACTIVE_SESSION_FILE
    with FileLock(str(marker.with_suffix(".lock")), timeout=5, mode=0o600):
        if hook == "sessionEnd":
            # Copilot detaches the old sessionEnd on /clear. A newer sessionStart
            # can win the race and must retain its marker.
            if (
                not marker.exists()
                or json.loads(marker.read_bytes()).get("copilot_session_id") != vendor_id
            ):
                return
            atomic_write_json(marker, {"copilot_session_id": None})
        else:
            atomic_write_json(marker, {"copilot_session_id": vendor_id})


def main() -> int:
    """Read Copilot's hook input from stdin and emit the vendor's JSON decision."""
    if len(sys.argv) != 4 or sys.argv[3] not in {*_LIFECYCLE_HOOKS, *_HOOKS}:
        return 2
    _, directory, session_id, hook = sys.argv
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("expected a hook object")
        if hook in _LIFECYCLE_HOOKS:
            _record_session(Path(directory), hook, payload)
            return 0
        result = evaluate(Path(directory), session_id, hook, payload)
    except (OSError, ValueError, TypeError, AttributeError):
        if hook in _LIFECYCLE_HOOKS:
            return 1
        result = _blocked(hook, "Omnigent could not read the tool policy request.")
    if hook == "postToolUseFailure" and (reason := result.get("additionalContext")):
        print(reason)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
