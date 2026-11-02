from __future__ import annotations

from typing import Protocol

from pipediag.diagnostics.explanation import ExplanationSpec
from pipediag.diagnostics.report import DiagnosticReportRenderer


class ExplanationGenerator(Protocol):
    """
    Interface for generating a human-readable explanation from
    a structured ExplanationSpec.

    Implementations must treat ExplanationSpec as the authoritative
    diagnostic information.

    The generator must not perform diagnosis, modify posterior values,
    introduce new evidence, or change threshold decisions.
    """

    def generate(
        self,
        explanation: ExplanationSpec,
    ) -> str:
        ...


class DeterministicExplanationGenerator:
    """
    Deterministic fallback explanation generator.

    This implementation does not require an LLM. It delegates directly
    to the existing deterministic DiagnosticReportRenderer.

    PipeDiag remains fully functional when no LLM provider is available.
    """

    def __init__(
        self,
        renderer: DiagnosticReportRenderer | None = None,
    ) -> None:
        self.renderer = (
            renderer
            if renderer is not None
            else DiagnosticReportRenderer()
        )

    def generate(
        self,
        explanation: ExplanationSpec,
    ) -> str:
        if explanation is None:
            raise ValueError(
                "explanation cannot be None"
            )

        report = self.renderer.render(explanation)

        if not isinstance(report, str):
            raise ValueError(
                "deterministic explanation generator must return a string"
            )

        return report
