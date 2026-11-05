from __future__ import annotations

from .diagnostic_result import DiagnosticResult
from .uncertainty import posterior_uncertainty


class DiagnosticEngine:
    def __init__(
        self,
        likelihood_evaluator,
        bayesian_evidence_inference,
    ):
        if likelihood_evaluator is None:
            raise ValueError(
                "likelihood_evaluator is required"
            )

        if bayesian_evidence_inference is None:
            raise ValueError(
                "bayesian_evidence_inference is required"
            )

        self.likelihood_evaluator = likelihood_evaluator
        self.bayesian_evidence_inference = (
            bayesian_evidence_inference
        )

    def diagnose(
        self,
        evidence_groups,
        metadata=None,
    ):
        if evidence_groups is None:
            raise ValueError(
                "evidence_groups is required"
            )

        seen_names: set[str] = set()

        evidence_log_ratios: dict[
            str,
            dict[str, float],
        ] = {}

        for evidence in evidence_groups:
            if evidence.name in seen_names:
                raise ValueError(
                    f"Duplicate evidence group "
                    f"'{evidence.name}'"
                )

            seen_names.add(evidence.name)

            log_ratios = (
                self.likelihood_evaluator
                .evaluate_log_ratios(evidence)
            )

            if log_ratios:
                evidence_log_ratios[
                    evidence.name
                ] = log_ratios


        evidence_supported_hypotheses = frozenset(
            hypothesis_id
            for log_ratios in evidence_log_ratios.values()
            for hypothesis_id in log_ratios
        )
        

        posterior = (
            self.bayesian_evidence_inference
            .infer_with_log_likelihood_ratios(
                evidence_log_ratios
            )
        )

        uncertainty = posterior_uncertainty(
            posterior
        )

        return DiagnosticResult(
            posterior=posterior,
            uncertainty=uncertainty,
            evidence=evidence_groups,
            metadata=metadata or {},
            evidence_supported_hypotheses=(
                evidence_supported_hypotheses
            ),
        )