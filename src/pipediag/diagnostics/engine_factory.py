from __future__ import annotations

from pathlib import Path

from .bayesian import BayesianInference
from .bayesian_evidence import BayesianEvidenceInference
from .calibrated_registry import load_calibrated_registry
from .diagnostic_engine import DiagnosticEngine
from .evidence_hypothesis_map import EVIDENCE_HYPOTHESIS_MAP
from .evidence_likelihood import EvidenceLikelihoodEvaluator
from .hypotheses import DIAGNOSTIC_HYPOTHESES
from .likelihood_registry import LikelihoodModelRegistry


def build_diagnostic_engine(
    registry: LikelihoodModelRegistry,
) -> DiagnosticEngine:
    """Build a diagnostic engine from a likelihood model registry."""
    if registry is None:
        raise ValueError("registry is required")

    hypothesis_ids = [
        hypothesis.id
        for hypothesis in DIAGNOSTIC_HYPOTHESES
    ]

    bayesian_inference = BayesianInference(
        hypothesis_ids=hypothesis_ids
    )

    likelihood_evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    bayesian_evidence_inference = BayesianEvidenceInference(
        bayesian_inference=bayesian_inference,
    )

    return DiagnosticEngine(
        likelihood_evaluator=likelihood_evaluator,
        bayesian_evidence_inference=bayesian_evidence_inference,
    )


def build_calibrated_diagnostic_engine(
    artifact_path: str | Path | None = None,
) -> DiagnosticEngine:
    """Build a production diagnostic engine from a calibrated artifact."""
    registry = load_calibrated_registry(artifact_path)
    return build_diagnostic_engine(registry)
