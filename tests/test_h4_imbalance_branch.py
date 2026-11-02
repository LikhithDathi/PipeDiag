import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import BayesianEvidenceInference
from pipediag.diagnostics.diagnostic_engine import DiagnosticEngine
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.hypotheses import DIAGNOSTIC_HYPOTHESES
from pipediag.diagnostics.likelihood_builder import LikelihoodModelBuilder

from experiments.imbalance_calibration import (
    collect_imbalance_calibration_observations,
)
from experiments.imbalance_evidence import collect_imbalance_evidence
from experiments.imbalance_pipeline import run_imbalance_pipeline


def _build_diagnostic_engine():
    observations = collect_imbalance_calibration_observations(
        n_samples=500,
        n_features=10,
        random_states=(1, 2, 3, 4, 5),
    )

    registry = LikelihoodModelBuilder().build(observations)

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    bayesian = BayesianInference(
        hypothesis_ids=[
            hypothesis.id
            for hypothesis in DIAGNOSTIC_HYPOTHESES
        ]
    )

    evidence_inference = BayesianEvidenceInference(bayesian)

    return DiagnosticEngine(
        likelihood_evaluator=evaluator,
        bayesian_evidence_inference=evidence_inference,
    )


def test_imbalance_evidence_has_correct_group():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    evidence = collect_imbalance_evidence(result)

    assert evidence.name == "G4"
    assert evidence.status == "observed"


def test_imbalance_evidence_uses_minority_proportion():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    evidence = collect_imbalance_evidence(result)

    assert evidence.value == pytest.approx(
        result.minority_class_proportion
    )


def test_imbalance_evidence_contains_measurements():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    evidence = collect_imbalance_evidence(result)

    measurements = evidence.metadata["measurements"]

    assert "minority_class_proportion" in measurements
    assert "imbalance_ratio" in measurements


def test_imbalance_evidence_preserves_class_counts():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    evidence = collect_imbalance_evidence(result)

    assert (
        evidence.metadata["train_class_counts"]
        == result.train_class_counts
    )

    assert (
        evidence.metadata["test_class_counts"]
        == result.test_class_counts
    )


def test_calibration_produces_observations():
    observations = collect_imbalance_calibration_observations()

    assert observations


def test_calibration_contains_true_and_false_samples():
    observations = collect_imbalance_calibration_observations()

    true_observations = [
        observation
        for observation in observations
        if observation.hypothesis_true
    ]

    false_observations = [
        observation
        for observation in observations
        if not observation.hypothesis_true
    ]

    assert true_observations
    assert false_observations


def test_calibration_uses_g4_h4():
    observations = collect_imbalance_calibration_observations()

    assert all(
        observation.evidence_group == "G4"
        for observation in observations
    )

    assert all(
        observation.hypothesis == "H4"
        for observation in observations
    )


def test_calibration_uses_correct_measurement():
    observations = collect_imbalance_calibration_observations()

    assert all(
        observation.measurement == "minority_class_proportion"
        for observation in observations
    )


def test_calibrated_model_is_registered():
    observations = collect_imbalance_calibration_observations()

    registry = LikelihoodModelBuilder().build(observations)

    model = registry.get("G4", "H4")

    assert model is not None
    assert model.measurement == "minority_class_proportion"


def test_h4_true_values_have_lower_minority_proportion():
    observations = collect_imbalance_calibration_observations()

    true_values = [
        observation.value
        for observation in observations
        if observation.hypothesis_true
    ]

    false_values = [
        observation.value
        for observation in observations
        if not observation.hypothesis_true
    ]

    assert sum(true_values) / len(true_values) < (
        sum(false_values) / len(false_values)
    )


def test_production_map_contains_h4():
    mapping = EVIDENCE_HYPOTHESIS_MAP["G4"]

    assert "H4" in mapping["primary"]


def test_evaluator_returns_h4_likelihood_ratio():
    observations = collect_imbalance_calibration_observations()

    registry = LikelihoodModelBuilder().build(observations)

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=100,
    )

    evidence = collect_imbalance_evidence(result)

    ratios = evaluator.evaluate_ratios(evidence)

    assert "H4" in ratios
    assert ratios["H4"] > 0


def test_h4_ratio_uses_minority_proportion():
    observations = collect_imbalance_calibration_observations()

    registry = LikelihoodModelBuilder().build(observations)

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=100,
    )

    evidence = collect_imbalance_evidence(result)

    model = registry.get("G4", "H4")

    expected_ratio = model.likelihood_ratio(
        evidence.metadata["measurements"][
            "minority_class_proportion"
        ].value
    )

    actual_ratio = evaluator.evaluate_ratios(evidence)["H4"]

    assert actual_ratio == pytest.approx(expected_ratio)


def test_strong_imbalance_increases_h4_support():
    engine = _build_diagnostic_engine()

    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=100,
    )

    evidence = collect_imbalance_evidence(result)

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert diagnostic.posterior["H4"] > 1.0 / 9.0


def test_less_imbalanced_data_has_weaker_h4_support():
    engine = _build_diagnostic_engine()

    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.40,
        random_state=100,
    )

    evidence = collect_imbalance_evidence(result)

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert diagnostic.posterior["H4"] < 0.5


def test_diagnostic_contains_all_hypotheses():
    engine = _build_diagnostic_engine()

    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=100,
    )

    evidence = collect_imbalance_evidence(result)

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    expected = {
        hypothesis.id
        for hypothesis in DIAGNOSTIC_HYPOTHESES
    }

    assert set(diagnostic.posterior) == expected


def test_diagnostic_is_normalized():
    engine = _build_diagnostic_engine()

    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=100,
    )

    evidence = collect_imbalance_evidence(result)

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert sum(diagnostic.posterior.values()) == pytest.approx(1.0)


def test_unavailable_g4_preserves_prior():
    engine = _build_diagnostic_engine()

    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=100,
    )

    evidence = collect_imbalance_evidence(result)

    unavailable = evidence.__class__(
        name=evidence.name,
        value=evidence.value,
        status="unavailable",
        metadata=evidence.metadata,
    )

    diagnostic = engine.diagnose(
        evidence_groups=[unavailable]
    )

    assert diagnostic.posterior["H4"] == pytest.approx(1.0 / 9.0)


def test_cleaner_class_distribution_has_less_imbalance():
    imbalanced = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    less_imbalanced = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.40,
        random_state=42,
    )

    assert (
        imbalanced.imbalance_ratio
        > less_imbalanced.imbalance_ratio
    )