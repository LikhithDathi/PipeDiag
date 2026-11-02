from __future__ import annotations

from typing import Any

from pipediag.application import DiagnosticRunResult

from .run_store import DiagnosticRunStore


class DiagnosticRunService:
    """Web-layer storage service; it does not perform diagnosis."""

    def __init__(
        self,
        store: DiagnosticRunStore | None = None,
    ) -> None:
        self.store = store if store is not None else DiagnosticRunStore()

    def submit(
        self,
        result: DiagnosticRunResult,
        runtime_context: Any | None = None,
    ) -> str:
        if result is None:
            raise ValueError("diagnostic result is required")

        run_id = result.diagnostic_result.metadata.get("run_id")
        if not run_id:
            raise ValueError(
                "diagnostic result metadata must contain run_id"
            )

        self.store.save(run_id, result, runtime_context)
        return run_id

    def get(self, run_id: str) -> DiagnosticRunResult | None:
        return self.store.get(run_id)

    def get_runtime_context(self, run_id: str) -> Any | None:
        return self.store.get_runtime_context(run_id)

    def delete(self, run_id: str) -> bool:
        return self.store.delete(run_id)

    def clear(self) -> None:
        self.store.clear()
