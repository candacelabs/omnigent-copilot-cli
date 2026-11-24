"""HTTP harness executor for the Copilot CLI already running in the session terminal."""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections.abc import AsyncIterator
from pathlib import Path

from omnigent.harnesses.copilot_native.bridge import (
    BRIDGE_DIR_ENV_VAR,
    REQUEST_SESSION_ID_ENV_VAR,
    CopilotSubmissionUncertain,
    bridge_dir_for_session_id,
    inject_interrupt,
    inject_user_message,
)
from omnigent.inner.executor import (
    EnqueuedContent,
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    InjectionDeliveryAmbiguous,
    Message,
    ToolSpec,
    TurnComplete,
)
from omnigent.inner.native_attachments import attachment_reference_line


class CopilotNativeExecutor(Executor):
    """Send new and steering messages to the session whose transcript the runner mirrors."""

    def __init__(self, bridge_dir: Path | None = None) -> None:
        if bridge_dir is None:
            session_id = os.environ.get(REQUEST_SESSION_ID_ENV_VAR)
            raw = os.environ.get(BRIDGE_DIR_ENV_VAR)
            if not raw or not session_id:
                raise RuntimeError(
                    f"copilot-native requires {BRIDGE_DIR_ENV_VAR} "
                    f"and {REQUEST_SESSION_ID_ENV_VAR}"
                )
            bridge_dir = Path(raw)
            if bridge_dir != bridge_dir_for_session_id(session_id):
                raise RuntimeError(
                    "Copilot's harness and terminal are bound to different sessions"
                )
        self._bridge_dir = bridge_dir
        self._inject_lock = asyncio.Lock()

    def supports_live_message_queue(self) -> bool:
        """Copilot accepts steering in its composer while a model turn is running."""
        return True

    async def _deliver(self, content: EnqueuedContent, model: str | None = None) -> None:
        text = _message_text(content, self._bridge_dir)
        async with self._inject_lock:
            cancel = threading.Event()
            task = asyncio.create_task(
                asyncio.to_thread(
                    inject_user_message,
                    self._bridge_dir,
                    content=text,
                    model=model,
                    cancel_event=cancel,
                )
            )
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # A cancelled HTTP request must not type later when a dialog closes.
                cancel.set()
                with contextlib.suppress(RuntimeError, OSError):
                    await task
                raise

    async def enqueue_session_message(self, session_key: str, content: EnqueuedContent) -> bool:
        del session_key
        try:
            await self._deliver(content)
        except CopilotSubmissionUncertain as exc:
            raise InjectionDeliveryAmbiguous(str(exc)) from exc
        except (RuntimeError, OSError):
            return False
        return True

    async def interrupt_session(self, session_key: str) -> bool:
        del session_key
        try:
            await asyncio.to_thread(inject_interrupt, self._bridge_dir)
        except (RuntimeError, OSError):
            return False
        return True

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        """The pane has the earlier turns. Only deliver the latest user input."""
        del tools, system_prompt
        content = next(
            (
                message.get("content")
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            None,
        )
        try:
            await self._deliver(content, config.model if config else None)
        except (RuntimeError, OSError) as exc:
            yield ExecutorError(
                message=str(exc),
                preserve_session=True,
                undelivered=not isinstance(exc, CopilotSubmissionUncertain),
            )
            return
        # The forwarder mirrors the vendor's authoritative user and assistant items.
        yield TurnComplete(response=None)


def _message_text(content: EnqueuedContent, bridge_dir: Path) -> str:
    if isinstance(content, str):
        return content
    attachments: list[str] = []
    text: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("input_image", "input_file"):
                attachments.append(attachment_reference_line(block, bridge_dir))
            elif block.get("type") in ("text", "input_text") and isinstance(
                part := block.get("text"), str
            ):
                text.append(part)
    return "\n\n".join(attachments + text)
