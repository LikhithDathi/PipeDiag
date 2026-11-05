
from __future__ import annotations

import math

from pipediag.diagnostics.multi_fault_bayesian import (
    MultiFaultBayesianInference,
)
from pipediag.diagnostics.multi_fault_diagnostic_result import (
    MultiFaultDiagnosticResult,
)


class MultiFaultDiagnosticEngine:
    """
    Coordinates normalized multi-fault Bayesian diagnosis.

    Independent Bayesian probabilities are used to determine
    which faults are detected.

    Normalized probabilities represent the relative diagnostic
    support among hypotheses with eligible evidence.
    """

    def __init__(
        self,
        likelihood_evaluator,
        bayesian_inference: MultiFaultBayesianInference,
        detection_threshold: float = 0.5,
    ):
        if likelihood_evaluator is None:
            raise ValueError(
                "likelihood_evaluator is required"
            )

        if bayesian_inference is None:
            raise ValueError(
                "bayesian_inference is required"
            )

        if not isinstance(
            bayesian_inference,
            MultiFaultBayesianInference,
        ):
            raise TypeError(
                "bayesian_inference must be a "
                "MultiFaultBayesianInference instance"
            )

        if (
            not isinstance(detection_threshold, (int, float))
            or not math.isfinite(detection_threshold)
            or not 0.0 < detection_threshold < 1.0
        ):
            raise ValueError(
                "detection_threshold must be between 0 and 1"
            )

        self.likelihood_evaluator = likelihood_evaluator
        self.bayesian_inference = bayesian_inference
        self.detection_threshold = float(detection_threshold)

    def diagnose(
        self,
        evidence_groups,
        metadata=None,
    ) -> MultiFaultDiagnosticResult:

        if evidence_groups is None:
            raise ValueError(
                "evidence_groups is required"
            )

        evidence_groups = list(evidence_groups)

        seen_names = set()
        evidence_log_ratios = {}

        for evidence in evidence_groups:

            if evidence.name in seen_names:
                raise ValueError(
                    f"Duplicate evidence group '{evidence.name}'"
                )

            seen_names.add(evidence.name)

            log_ratios = (
                self.likelihood_evaluator
                .evaluate_log_ratios(evidence)
            )

            if log_ratios:
                evidence_log_ratios[evidence.name] = log_ratios

        evidence_supported_hypotheses = frozenset(
            hypothesis
            for ratios in evidence_log_ratios.values()
            for hypothesis in ratios
        )

        # Independent probabilities are used for detection.
        independent_posterior = (
            self.bayesian_inference
            .infer_independent_probabilities(
                evidence_log_ratios
            )
        )

        # Normalized probabilities are used for reporting.
        posterior = (
            self.bayesian_inference
            .infer_with_log_likelihood_ratios(
                evidence_log_ratios
            )
        )

        uncertainty_by_hypothesis = {
            hypothesis: self._binary_entropy(probability)
            for hypothesis, probability
            in independent_posterior.items()
        }

        detected_hypotheses = frozenset(
            hypothesis
            for hypothesis, probability
            in independent_posterior.items()
            if (
                probability >= self.detection_threshold
                and hypothesis in evidence_supported_hypotheses
            )
        )

        return MultiFaultDiagnosticResult(
            posterior=posterior,
            independent_posterior=independent_posterior,
            uncertainty_by_hypothesis=uncertainty_by_hypothesis,
            detected_hypotheses=detected_hypotheses,
            evidence=evidence_groups,
            evidence_supported_hypotheses=(
                evidence_supported_hypotheses
            ),
            evidence_log_likelihood_ratios=evidence_log_ratios,
            detection_threshold=self.detection_threshold,
            metadata=metadata or {},
        )

    @staticmethod
    def _binary_entropy(probability: float) -> float:
        """
        Normalized binary entropy of an independent probability.

        0.0 means maximum certainty.
        1.0 means maximum uncertainty.
        """

        if probability <= 0.0 or probability >= 1.0:
            return 0.0

        return -(
            probability * math.log2(probability)
            + (1.0 - probability)
            * math.log2(1.0 - probability)
        )
