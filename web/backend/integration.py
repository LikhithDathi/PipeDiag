from __future__ import annotations

from typing import Any

from pipediag.application import DiagnosticRunResult

from .service import DiagnosticRunService


class DiagnosticWebIntegration:
    """Bridge completed PipeDiag runs into the web layer."""

    def __init__(
        self,
        service: DiagnosticRunService | None = None,
    ) -> None:
        self.service = (
            service if service is not None else DiagnosticRunService()
        )

    def publish(
        self,
        result: DiagnosticRunResult,
        runtime_context: Any | None = None,
    ) -> str:
        if result is None:
            raise ValueError("diagnostic result is required")

        return self.service.submit(result, runtime_context)
