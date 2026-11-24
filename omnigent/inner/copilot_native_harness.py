"""Omnigent HTTP harness for the native GitHub Copilot CLI terminal."""

from __future__ import annotations

from fastapi import FastAPI

from omnigent.inner.copilot_native_executor import CopilotNativeExecutor
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter


def create_app() -> FastAPI:
    """Build the HTTP-to-executor adapter used by the isolated harness worker."""
    return ExecutorAdapter(executor_factory=CopilotNativeExecutor).build()
