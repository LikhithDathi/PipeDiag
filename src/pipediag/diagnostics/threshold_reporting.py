from __future__ import annotations

from dataclasses import dataclass

from pipediag.diagnostics.diagnostic_result import DiagnosticResult


DEFAULT_REPORTING_THRESHOLD = 0.20


@dataclass(frozen=True)
class ReportedHypothesis:
    hypothesis_id: str
    posterior: float

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError(
                "ReportedHypothesis requires a non-empty hypothesis_id"
            )

        if not 0.0 <= self.posterior <= 1.0:
            raise ValueError(
                "ReportedHypothesis posterior must be between 0 and 1"
            )


@dataclass(frozen=True)
class ThresholdDiagnosticReport:
    threshold: float
    reported_hypotheses: tuple[ReportedHypothesis, ...]
    uncertainty: float

    @property
    def is_inconclusive(self) -> bool:
        return not self.reported_hypotheses

    @property
    def reported_hypothesis_ids(self) -> tuple[str, ...]:
        return tuple(
            hypothesis.hypothesis_id
            for hypothesis in self.reported_hypotheses
        )


class ThresholdReporter:
    def __init__(
        self,
        threshold: float = DEFAULT_REPORTING_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

        self.threshold = threshold

    def report(
        self,
        diagnostic_result: DiagnosticResult,
    ) -> ThresholdDiagnosticReport:
        if diagnostic_result is None:
            raise ValueError(
                "diagnostic_result cannot be None"
            )

        reported = [
            ReportedHypothesis(
                hypothesis_id=hypothesis_id,
                posterior=posterior,
            )
            for hypothesis_id, posterior
            in diagnostic_result.posterior.items()
            if posterior >= self.threshold
        ]

        reported.sort(
            key=lambda hypothesis: hypothesis.posterior,
            reverse=True,
        )

        return ThresholdDiagnosticReport(
            threshold=self.threshold,
            reported_hypotheses=tuple(reported),
            uncertainty=diagnostic_result.uncertainty,
        )