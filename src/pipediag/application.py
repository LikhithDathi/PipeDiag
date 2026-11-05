from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.explanation import ExplanationBuilder, ExplanationSpec
from pipediag.diagnostics.report import DiagnosticReportRenderer

from pipediag.llm.generator import (
    ExplanationGenerator,
    DeterministicExplanationGenerator,
)

from pipediag.diagnostics.multi_fault_diagnostic_result import (
    MultiFaultDiagnosticResult,
)

class EvidenceProvider(Protocol):
    """
    Protocol for components that acquire diagnostic evidence.

    The application layer intentionally knows nothing about how evidence
    is collected. It only requires a provider with a collect() method.
    """

    def collect(self, source: Any) -> list:
        ...


class DiagnosticEvidenceProvider:
    """
    Adapts a provider of DiagnosticEvidence into EvidenceGroup objects
    consumable by the DiagnosticApplication.
    """

    def __init__(self, provider):
        if provider is None:
            raise ValueError("provider is required")

        self.provider = provider

    def collect(self, source):
        evidence = self.provider.collect(source)

        if evidence is None:
            raise ValueError("evidence provider returned None")

        from pipediag.diagnostics.adapters import diagnostic_evidence_to_group

        return [
            diagnostic_evidence_to_group(item)
            for item in evidence
        ]

class DiagnosticEngineProtocol(Protocol):
    """
    Minimal interface required from the diagnostic engine.
    """

    def diagnose(
        self,
        evidence_groups,
        metadata=None,
    ) -> DiagnosticResult:
        ...


@dataclass(frozen=True)
class DiagnosticRunResult:
    """
    Complete result of one end-to-end diagnostic run.

    The diagnostic_result contains the probabilistic inference,
    explanation contains the deterministic structured interpretation,
    and report contains the final human-readable representation.
    """

    diagnostic_result: DiagnosticResult
    explanation: ExplanationSpec
    report: str
    multi_fault_result: MultiFaultDiagnosticResult | None = None

    def __post_init__(self) -> None:
        if self.diagnostic_result is None:
            raise ValueError(
                "DiagnosticRunResult requires a diagnostic_result"
            )

        if self.explanation is None:
            raise ValueError(
                "DiagnosticRunResult requires an explanation"
            )

        if not isinstance(self.report, str):
            raise ValueError(
                "DiagnosticRunResult report must be a string"
            )


class DiagnosticApplication:
    """
    High-level orchestration layer for the diagnostic framework.

    The application coordinates:

        evidence acquisition
            -> Bayesian diagnosis
            -> deterministic explanation
            -> report rendering

    It does not implement any diagnostic logic itself.
    """

    def __init__(
        self,
        *,
        evidence_provider: EvidenceProvider,
        diagnostic_engine: DiagnosticEngineProtocol,
        explanation_builder: ExplanationBuilder | None = None,
        report_renderer: DiagnosticReportRenderer | None = None,
        explanation_generator: ExplanationGenerator | None = None,
    ) -> None:
        if evidence_provider is None:
            raise ValueError(
                "evidence_provider cannot be None"
            )

        if diagnostic_engine is None:
            raise ValueError(
                "diagnostic_engine cannot be None"
            )

        if isinstance(evidence_provider, DiagnosticEvidenceProvider):
            self.evidence_provider = evidence_provider
        else:
            self.evidence_provider = evidence_provider


        self.diagnostic_engine = diagnostic_engine

        self.explanation_builder = (
            explanation_builder
            if explanation_builder is not None
            else ExplanationBuilder()
        )

        self.report_renderer = (
            report_renderer
            if report_renderer is not None
            else DiagnosticReportRenderer()
        )

        self.explanation_generator = (
            explanation_generator
            if explanation_generator is not None
            else DeterministicExplanationGenerator(
                renderer=self.report_renderer
            )
        )

    def run(
        self,
        source: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosticRunResult:
        """
        Execute the complete diagnostic workflow.

        Parameters
        ----------
        source:
            Input understood by the configured evidence provider.

        metadata:
            Optional metadata attached to the diagnostic result.

        Returns
        -------
        DiagnosticRunResult
            Contains the diagnostic result, structured explanation,
            and rendered report.
        """
        evidence = self.evidence_provider.collect(source)

        if evidence is None:
            raise ValueError(
                "Evidence provider returned None"
            )

        diagnostic_result = self.diagnostic_engine.diagnose(
            evidence_groups=evidence,
            metadata=metadata,
        )

        explanation = self.explanation_builder.build(
            diagnostic_result
        )

        report = self.explanation_generator.generate(
            explanation
        )

        return DiagnosticRunResult(
            diagnostic_result=diagnostic_result,
            explanation=explanation,
            report=report,
        )
