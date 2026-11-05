from dataclasses import dataclass, field
import math
from typing import Any

from pipediag.diagnostics.evidence_group import EvidenceGroup


@dataclass(frozen=True)
class DiagnosticResult:
    posterior: dict[str, float]
    uncertainty: float
    evidence: list[EvidenceGroup]
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_supported_hypotheses: frozenset[str] | None = None

    def __post_init__(self):
        if not self.posterior:
            raise ValueError(
                "DiagnosticResult requires a non-empty posterior"
            )

        for probability in self.posterior.values():
            if not math.isfinite(probability):
                raise ValueError(
                    "Posterior probabilities must be finite"
                )

            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "Posterior probabilities must be between 0 and 1"
                )

        if not math.isclose(
            sum(self.posterior.values()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Posterior probabilities must sum to one"
            )

        if not math.isfinite(self.uncertainty):
            raise ValueError(
                "Uncertainty must be finite"
            )

        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(
                "Uncertainty must be between 0 and 1"
            )