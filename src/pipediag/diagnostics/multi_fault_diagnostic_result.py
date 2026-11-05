
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from pipediag.diagnostics.evidence_group import EvidenceGroup


@dataclass(frozen=True)
class MultiFaultDiagnosticResult:
    """
    Result of normalized multi-fault diagnosis.

    posterior:
        Normalized diagnostic support distribution. Its values
        sum to one across hypotheses with eligible evidence.

    independent_posterior:
        Independent Bayesian probabilities used for fault detection.
        These probabilities do not need to sum to one.
    """

    posterior: dict[str, float]

    uncertainty_by_hypothesis: dict[str, float]

    detected_hypotheses: frozenset[str]

    evidence: list[EvidenceGroup]

    evidence_supported_hypotheses: frozenset[str]

    evidence_log_likelihood_ratios: dict[
        str,
        dict[str, float],
    ] = field(default_factory=dict)

    detection_threshold: float = 0.5

    metadata: dict[str, Any] = field(default_factory=dict)

    independent_posterior: dict[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self):

        if not self.posterior:
            raise ValueError(
                "MultiFaultDiagnosticResult requires a non-empty posterior"
            )

        posterior_total = sum(self.posterior.values())

        if not math.isclose(
            posterior_total,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Normalized posterior probabilities must sum to one"
            )

        if set(self.posterior) != set(
            self.uncertainty_by_hypothesis
        ):
            raise ValueError(
                "Posterior and uncertainty hypotheses must match"
            )

        # Backward compatibility for callers constructing the result
        # directly without supplying independent probabilities.
        if not self.independent_posterior:
            object.__setattr__(
                self,
                "independent_posterior",
                dict(self.posterior),
            )

        if set(self.posterior) != set(
            self.independent_posterior
        ):
            raise ValueError(
                "Normalized and independent posterior hypotheses "
                "must match"
            )

        for hypothesis, probability in self.posterior.items():

            if not math.isfinite(probability):
                raise ValueError(
                    "Posterior probabilities must be finite"
                )

            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "Posterior probabilities must be between 0 and 1"
                )

            independent_probability = (
                self.independent_posterior[hypothesis]
            )

            if (
                not math.isfinite(independent_probability)
                or not 0.0 <= independent_probability <= 1.0
            ):
                raise ValueError(
                    "Independent probabilities must be finite "
                    "and between 0 and 1"
                )

            uncertainty = self.uncertainty_by_hypothesis[
                hypothesis
            ]

            if (
                not math.isfinite(uncertainty)
                or not 0.0 <= uncertainty <= 1.0
            ):
                raise ValueError(
                    "Hypothesis uncertainty must be between 0 and 1"
                )

        if (
            not math.isfinite(self.detection_threshold)
            or not 0.0 < self.detection_threshold < 1.0
        ):
            raise ValueError(
                "Detection threshold must be between 0 and 1"
            )

        unknown_detected = (
            set(self.detected_hypotheses)
            - set(self.posterior)
        )

        if unknown_detected:
            raise ValueError(
                "Detected hypotheses must exist in posterior"
            )

    @property
    def posterior_sum(self) -> float:
        return sum(self.posterior.values())

    @property
    def uncertainty(self) -> float:
        """
        Mean binary entropy of independent hypothesis probabilities.
        """

        if not self.uncertainty_by_hypothesis:
            return 0.0

        return (
            sum(self.uncertainty_by_hypothesis.values())
            / len(self.uncertainty_by_hypothesis)
        )

    @property
    def expected_hypothesis(self) -> None:
        """Multi-fault results have no single expected hypothesis."""

        return None

    @property
    def ranked_hypotheses(self) -> list[tuple[str, float]]:
        return sorted(
            self.posterior.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    @property
    def undetected_hypotheses(self) -> frozenset[str]:
        return frozenset(
            set(self.posterior)
            - set(self.detected_hypotheses)
        )
