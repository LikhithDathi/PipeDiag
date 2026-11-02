import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.diagnostic_engine import (
    DiagnosticEngine,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.hypotheses import (
    DIAGNOSTIC_HYPOTHESES,
)
from pipediag.diagnostics.likelihood_builder import (
    LikelihoodModelBuilder,
)

from experiments.generalization_calibration import (
    collect_generalization_calibration_observations,
)
from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import (
    run_overfitting_pipeline,
)


CALIBRATED_MAP = {
    "G2": {
        "primary": {"H2"},
        "secondary": set(),
    }
}


def _build_diagnostic_engine():
    observations = (
        collect_generalization_calibration_observations(
            n_samples=500,
            n_features=10,
            random_states=(1, 2, 3, 4, 5),
        )
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=CALIBRATED_MAP,
    )

    bayesian = BayesianInference(
        hypothesis_ids=[
            hypothesis.id
            for hypothesis in DIAGNOSTIC_HYPOTHESES
        ]
    )

    evidence_inference = BayesianEvidenceInference(
        bayesian
    )

    return DiagnosticEngine(
        likelihood_evaluator=evaluator,
        bayesian_evidence_inference=evidence_inference,
    )


def test_strong_generalization_gap_increases_h2_support():
    engine = _build_diagnostic_engine()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert diagnostic.posterior["H2"] > 1.0 / 9.0


def test_small_generalization_gap_reduces_h2_support():
    engine = _build_diagnostic_engine()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=2,
    )

    evidence = collect_generalization_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert diagnostic.posterior["H2"] < 1.0 / 9.0


def test_h2_diagnostic_contains_all_hypotheses():
    engine = _build_diagnostic_engine()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    expected = {
        hypothesis.id
        for hypothesis in DIAGNOSTIC_HYPOTHESES
    }

    assert set(diagnostic.posterior) == expected


def test_h2_diagnostic_is_normalized():
    engine = _build_diagnostic_engine()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert sum(
        diagnostic.posterior.values()
    ) == pytest.approx(1.0)


def test_unavailable_g2_preserves_prior():
    engine = _build_diagnostic_engine()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(
        result
    )

    unavailable = evidence.__class__(
        name=evidence.name,
        value=evidence.value,
        status="unavailable",
        metadata=evidence.metadata,
    )

    diagnostic = engine.diagnose(
        evidence_groups=[unavailable]
    )

    assert diagnostic.posterior["H2"] == pytest.approx(
        1.0 / 9.0
    )