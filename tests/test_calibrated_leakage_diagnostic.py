import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.diagnostic_engine import DiagnosticEngine
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.hypotheses import DIAGNOSTIC_HYPOTHESES
from pipediag.diagnostics.likelihood_builder import (
    LikelihoodModelBuilder,
)

from experiments.leakage_calibration import (
    collect_leakage_calibration_observations,
)
from experiments.leakage_evidence import (
    collect_leakage_evidence,
)
from experiments.leakage_pipeline import (
    run_leakage_pipeline,
)

from experiments.clean_pipeline import run_clean_pipeline


CALIBRATED_MAP = {
    "G1": {
        "primary": {"H1"},
        "secondary": set(),
    }
}


def _build_diagnostic_engine():
    observations = collect_leakage_calibration_observations(
        n_samples=500,
        n_features=10,
        random_states=(1, 2, 3, 4, 5),
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


def test_clean_pipeline_does_not_strongly_support_leakage():
    engine = _build_diagnostic_engine()

    pipeline_result = run_clean_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert result.posterior["H1"] < 0.5


def test_strong_leakage_increases_h1_support():
    engine = _build_diagnostic_engine()

    pipeline_result = run_leakage_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        leakage_strength=1.0,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert result.posterior["H1"] > 1.0 / 9.0


def test_diagnostic_result_contains_all_hypotheses():
    engine = _build_diagnostic_engine()

    pipeline_result = run_leakage_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        leakage_strength=1.0,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

    expected = {
        hypothesis.id
        for hypothesis in DIAGNOSTIC_HYPOTHESES
    }

    assert set(result.posterior) == expected


def test_diagnostic_result_is_normalized():
    engine = _build_diagnostic_engine()

    pipeline_result = run_leakage_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        leakage_strength=1.0,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert sum(result.posterior.values()) == pytest.approx(
        1.0
    )


def test_unavailable_leakage_evidence_preserves_prior():
    engine = _build_diagnostic_engine()

    pipeline_result = run_leakage_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        leakage_strength=1.0,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    unavailable = evidence.__class__(
        name=evidence.name,
        value=evidence.value,
        status="unavailable",
        metadata=evidence.metadata,
    )

    result = engine.diagnose(
        evidence_groups=[unavailable]
    )

    assert result.posterior["H1"] == pytest.approx(
        1.0 / 9.0
    )