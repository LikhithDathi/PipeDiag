from __future__ import annotations

from threading import Lock
from typing import Any

from pipediag.application import DiagnosticRunResult


class DiagnosticRunStore:
    """In-memory storage for diagnostic results and runtime contexts."""

    def __init__(self) -> None:
        self._runs: dict[str, DiagnosticRunResult] = {}
        self._runtime_contexts: dict[str, Any] = {}
        self._lock = Lock()

    def save(
        self,
        run_id: str,
        result: DiagnosticRunResult,
        runtime_context: Any | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        if result is None:
            raise ValueError("result is required")

        with self._lock:
            self._runs[run_id] = result
            if runtime_context is None:
                self._runtime_contexts.pop(run_id, None)
            else:
                self._runtime_contexts[run_id] = runtime_context

    def get(self, run_id: str) -> DiagnosticRunResult | None:
        if not run_id:
            raise ValueError("run_id is required")
        with self._lock:
            return self._runs.get(run_id)

    def get_runtime_context(self, run_id: str) -> Any | None:
        if not run_id:
            raise ValueError("run_id is required")
        with self._lock:
            return self._runtime_contexts.get(run_id)

    def exists(self, run_id: str) -> bool:
        return self.get(run_id) is not None

    def delete(self, run_id: str) -> bool:
        if not run_id:
            raise ValueError("run_id is required")
        with self._lock:
            removed = self._runs.pop(run_id, None) is not None
            self._runtime_contexts.pop(run_id, None)
            return removed

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
            self._runtime_contexts.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._runs)
