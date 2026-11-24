"""Session-local Copilot command hooks use the runner's authenticated policy relay."""

from __future__ import annotations

import json
import stat
import sys
from io import StringIO
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from omnigent.harnesses.copilot_native import hook
from omnigent.harnesses.copilot_native.bridge import copilot_session_id_for_session_id
from omnigent.native.native_policy_hook import PolicyHookEvaluationRequest

_VENDOR_ID = copilot_session_id_for_session_id("conv_test")


def _relay(
    directory: Path, *, session_id: str = "conv_test", url: str = "http://127.0.0.1:4901"
) -> None:
    (directory / "tool_relay.json").write_text(
        json.dumps(
            {
                "url": url,
                "session_id": session_id,
                "token": "local-fixture-token",
            }
        )
    )


def test_scoped_hook_manifest_tracks_only_foreground_vendor_sessions(tmp_path: Path) -> None:
    bridge = hook.prepare_policy_plugin(tmp_path / "bridge", session_id="conv_test")
    manifest = json.loads((bridge / "plugin.json").read_bytes())
    hooks = json.loads((bridge / manifest["hooks"]).read_bytes())["hooks"]
    assert set(hooks) == {
        "sessionStart",
        "sessionEnd",
        "preToolUse",
        "postToolUse",
        "postToolUseFailure",
    }
    assert stat.S_IMODE(bridge.stat().st_mode) == 0o700
    for filename in ("plugin.json", "hooks.json"):
        assert stat.S_IMODE((bridge / filename).stat().st_mode) == 0o600
    for name, [entry] in hooks.items():
        assert entry["exec"] == sys.executable
        assert entry["args"] == ["-Im", hook.__name__, str(bridge), "conv_test", name]
    first, second = str(uuid4()), str(uuid4())
    marker = bridge / hook.ACTIVE_SESSION_FILE
    hook._record_session(bridge, "sessionStart", {"source": "startup", "sessionId": first})
    hook._record_session(bridge, "sessionStart", {"agentId": "child", "sessionId": second})
    assert json.loads(marker.read_bytes()) == {"copilot_session_id": first}
    hook._record_session(bridge, "sessionEnd", {"reason": "user_exit", "sessionId": first})
    assert json.loads(marker.read_bytes()) == {"copilot_session_id": None}
    hook._record_session(bridge, "sessionStart", {"source": "new", "sessionId": second})
    hook._record_session(bridge, "sessionEnd", {"reason": "user_exit", "sessionId": first})
    assert json.loads(marker.read_bytes()) == {"copilot_session_id": second}


def test_tool_and_result_decisions_use_copilot_fields_and_the_local_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _relay(tmp_path)
    evaluations: list[PolicyHookEvaluationRequest] = []
    verdict: dict[str, str] = {"result": "POLICY_ACTION_ALLOW"}

    def post(
        url: str,
        headers: dict[str, str],
        request: PolicyHookEvaluationRequest,
        timeout: int,
        label: str,
    ) -> tuple[httpx.Response, None]:
        assert url == "http://127.0.0.1:4901/policies/evaluate"
        assert headers == {"Authorization": "Bearer local-fixture-token"}
        evaluations.append(request)
        return httpx.Response(200, json=verdict), None

    monkeypatch.setattr(hook, "post_evaluate_with_retry", post)
    args = {"path": "note.txt", "file_text": "hello"}
    payload = {"sessionId": _VENDOR_ID, "toolName": "create", "toolArgs": json.dumps(args)}
    assert hook.evaluate(tmp_path, "conv_test", "preToolUse", payload) == {}
    assert evaluations[-1]["event"] == {
        "type": "PHASE_TOOL_CALL",
        "target": "",
        "data": {"name": "create", "arguments": args},
        "context": {"harness": "copilot-native"},
    }
    verdict.update(result="POLICY_ACTION_DENY", reason="Private output")
    assert hook.evaluate(tmp_path, "conv_test", "preToolUse", payload) == {
        "permissionDecision": "deny",
        "permissionDecisionReason": "Private output",
    }
    original = {"resultType": "success", "textResultForLlm": "Created the file."}
    assert hook.evaluate(
        tmp_path, "conv_test", "postToolUse", {**payload, "toolResult": original}
    ) == {
        "modifiedResult": {"resultType": "success", "textResultForLlm": "Private output"},
    }
    event = evaluations[-1]["event"]
    assert event["type"] == "PHASE_TOOL_RESULT"
    assert event["data"] == {"result": original}
    assert event["request_data"] == {"name": "create", "arguments": args}
    verdict.clear()
    assert (
        hook.evaluate(tmp_path, "conv_test", "preToolUse", payload)["permissionDecision"] == "deny"
    )


@pytest.mark.parametrize(
    ("session_id", "url"),
    [("different-session", "http://127.0.0.1:8910"), ("conv_test", "https://example.invalid")],
)
def test_reject_unrelated_relays(tmp_path: Path, session_id: str, url: str) -> None:
    _relay(tmp_path, session_id=session_id, url=url)
    assert hook.evaluate(
        tmp_path, "conv_test", "preToolUse", {"sessionId": _VENDOR_ID, "toolName": "view"}
    ) == {
        "permissionDecision": "deny",
        "permissionDecisionReason": "Omnigent's session policy relay is unavailable.",
    }


def test_only_declared_session_relay_tools_skip_the_second_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _relay(tmp_path)
    path = tmp_path / "tool_relay.json"
    manifest = json.loads(path.read_bytes())
    path.write_text(json.dumps({**manifest, "tools": [{"name": "lookup_fixture"}]}))
    monkeypatch.setattr(hook, "post_evaluate_with_retry", lambda *args: (None, "no policy"))
    payload = {"sessionId": _VENDOR_ID, "toolName": "omnigent-lookup_fixture"}
    assert hook.evaluate(tmp_path, "conv_test", "preToolUse", payload) == {}
    for name in (
        "omnigent-unknown",
        "github-lookup_fixture",
        "mcp__omnigent__other-lookup",
        "view",
    ):
        assert (
            hook.evaluate(tmp_path, "conv_test", "preToolUse", {**payload, "toolName": name})[
                "permissionDecision"
            ]
            == "deny"
        )
    other = {**payload, "sessionId": str(uuid4())}
    assert (
        hook.evaluate(tmp_path, "conv_test", "preToolUse", other)["permissionDecision"] == "deny"
    )
    child = {**other, "agentId": "worker"}
    assert (
        hook.evaluate(tmp_path, "conv_test", "preToolUse", child)["permissionDecision"] == "deny"
    )
    hook._record_session(tmp_path, "sessionStart", {"sessionId": _VENDOR_ID})
    assert hook.evaluate(tmp_path, "conv_test", "preToolUse", child) == {}


def test_failure_hook_uses_vendor_feedback_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys, "argv", [hook.__file__, str(tmp_path), "conv_test", "postToolUseFailure"]
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        StringIO(json.dumps({"sessionId": _VENDOR_ID, "toolName": "view", "error": "Failed"})),
    )
    assert hook.main() == 2
    assert capsys.readouterr().out.strip() == "Omnigent's session policy relay is unavailable."
