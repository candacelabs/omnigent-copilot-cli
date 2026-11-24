"""The web executor queues into the resident TUI and leaves output to its forwarder."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import Mock

import pytest

from omnigent.harnesses.copilot_native import bridge
from omnigent.inner import copilot_native_executor as native
from omnigent.inner.copilot_native_harness import create_app
from omnigent.inner.executor import (
    ExecutorConfig,
    ExecutorError,
    InjectionDeliveryAmbiguous,
    TurnComplete,
)


async def test_web_turn_uses_latest_message_and_materialized_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inject = Mock()
    monkeypatch.setattr(native, "inject_user_message", inject)
    executor = native.CopilotNativeExecutor(tmp_path)
    payload = base64.b64encode(b"chart contents").decode()
    messages = [
        {"role": "user", "content": "older question"},
        {"role": "assistant", "content": "older answer"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Read the attachment"},
                {
                    "type": "input_file",
                    "filename": "chart.txt",
                    "file_data": f"data:text/plain;base64,{payload}",
                },
            ],
        },
    ]
    assert not executor.supports_streaming()
    assert executor.supports_live_message_queue()
    result = [
        e async for e in executor.run_turn(messages, [], "", ExecutorConfig(model="test-model"))
    ]
    assert result == [TurnComplete(response=None)]
    text = inject.call_args.kwargs["content"]
    marker, question = text.split("\n\n")
    assert question == "Read the attachment"
    assert marker.startswith("[Attached: ") and marker.endswith("]")
    assert Path(marker[len("[Attached: ") : -1]).read_bytes() == b"chart contents"
    assert inject.call_args.kwargs["model"] == "test-model"
    assert await executor.enqueue_session_message("session", "and the next item")
    assert inject.call_args.kwargs["content"] == "and the next item"


@pytest.mark.parametrize("ambiguous", [False, True])
async def test_preserve_pane_and_do_not_retry_uncertain_submissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ambiguous: bool
) -> None:
    error = (
        bridge.CopilotSubmissionUncertain("could not confirm")
        if ambiguous
        else RuntimeError("terminal has a draft")
    )
    monkeypatch.setattr(native, "inject_user_message", Mock(side_effect=error))
    executor = native.CopilotNativeExecutor(tmp_path)
    result = [e async for e in executor.run_turn([{"role": "user", "content": "hi"}], [], "")]
    assert len(result) == 1 and isinstance(result[0], ExecutorError)
    assert result[0].message == str(error)
    assert result[0].preserve_session and result[0].undelivered != ambiguous
    if ambiguous:
        with pytest.raises(InjectionDeliveryAmbiguous):
            await executor.enqueue_session_message("session", "hi")
    else:
        assert not await executor.enqueue_session_message("session", "hi")


def test_harness_uses_runner_bound_bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(bridge.BRIDGE_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(bridge.REQUEST_SESSION_ID_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="copilot-native requires"):
        native.CopilotNativeExecutor()
    monkeypatch.setenv(bridge.BRIDGE_DIR_ENV_VAR, str(tmp_path))
    monkeypatch.setenv(bridge.REQUEST_SESSION_ID_ENV_VAR, "session-a")
    with pytest.raises(RuntimeError, match="different sessions"):
        native.CopilotNativeExecutor()
    monkeypatch.setenv(
        bridge.BRIDGE_DIR_ENV_VAR, str(bridge.bridge_dir_for_session_id("session-a"))
    )
    assert native.CopilotNativeExecutor().supports_live_message_queue()
    assert create_app().title
