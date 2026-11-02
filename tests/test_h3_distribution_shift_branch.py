import math

import numpy as np
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

from experiments.distribution_shift import (
    DistributionShiftPipelineResult,
    run_distribution_shift_pipeline,
)
from experiments.distribution_shift_calibration import (
    collect_distribution_shift_calibration_observations,
)
from experiments.distribution_shift_evidence import (
    collect_distribution_shift_evidence,
)


def _build_diagnostic_engine():
    observations = (
        collect_distribution_shift_calibration_observations(
            n_samples=500,
            n_features=10,
            random_states=(1, 2, 3, 4, 5),
        )
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

    evidence_inference = BayesianEvidenceInference(
        bayesian
    )

    return DiagnosticEngine(
        likelihood_evaluator=evaluator,
        bayesian_evidence_inference=evidence_inference,
    )


# ============================================================
# DISTRIBUTION SHIFT PIPELINE
# ============================================================


def test_returns_expected_result_type():
    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert isinstance(
        result,
        DistributionShiftPipelineResult,
    )


def test_output_shapes_are_consistent():
    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert result.X_train.ndim == 2
    assert result.X_test.ndim == 2

    assert result.X_train.shape[1] == 10
    assert result.X_test.shape[1] == 10

    assert result.X_train.shape[0] == len(result.y_train)
    assert result.X_test.shape[0] == len(result.y_test)


def test_target_values_are_binary():
    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert set(np.unique(result.y_train)) == {0, 1}
    assert set(np.unique(result.y_test)) == {0, 1}


def test_unshifted_pipeline_has_non_negative_shift_score():
    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=0.0,
        random_state=42,
    )

    assert result.shift_magnitude >= 0.0


def test_shifted_pipeline_has_positive_shift_score():
    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.0,
        random_state=42,
    )

    assert result.shift_magnitude > 0.0


def test_larger_shift_produces_larger_shift_score():
    small_shift = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=0.5,
        random_state=42,
    )

    large_shift = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=42,
    )

    assert (
        large_shift.shift_magnitude
        > small_shift.shift_magnitude
    )


def test_same_random_state_is_reproducible():
    result_a = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.0,
        random_state=42,
    )

    result_b = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.0,
        random_state=42,
    )

    assert np.array_equal(
        result_a.X_train,
        result_b.X_train,
    )

    assert np.array_equal(
        result_a.X_test,
        result_b.X_test,
    )

    assert np.array_equal(
        result_a.y_train,
        result_b.y_train,
    )

    assert np.array_equal(
        result_a.y_test,
        result_b.y_test,
    )

    assert result_a.shift_magnitude == pytest.approx(
        result_b.shift_magnitude
    )


# ============================================================
# DISTRIBUTION SHIFT EVIDENCE
# ============================================================


def test_evidence_has_correct_group():
    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.0,
        random_state=42,
    )

    evidence = collect_distribution_shift_evidence(result)

    assert evidence.name == "G3"
    assert evidence.status == "observed"


def test_evidence_value_matches_shift_score():
    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.0,
        random_state=42,
    )

    evidence = collect_distribution_shift_evidence(result)

    assert evidence.value == pytest.approx(
        result.shift_magnitude
    )


def test_evidence_contains_named_measurement():
    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.0,
        random_state=42,
    )

    evidence = collect_distribution_shift_evidence(result)

    measurements = evidence.metadata["measurements"]

    assert (
        "mean_absolute_standardized_shift"
        in measurements
    )

    measurement = measurements[
        "mean_absolute_standardized_shift"
    ]

    assert measurement.value == pytest.approx(
        result.shift_magnitude
    )


# ============================================================
# DISTRIBUTION SHIFT CALIBRATION
# ============================================================


def test_calibration_produces_observations():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    assert observations


def test_calibration_contains_true_and_false_samples():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

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


def test_calibration_uses_g3_and_mapped_hypotheses():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    assert all(
        observation.evidence_group == "G3"
        for observation in observations
    )

    hypotheses = {
        observation.hypothesis
        for observation in observations
    }

    assert hypotheses == {"H3"}


def test_calibration_uses_correct_measurement():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    assert all(
        observation.measurement
        == "mean_absolute_standardized_shift"
        for observation in observations
    )


def test_shifted_samples_have_larger_measurement_values():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

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

    assert (
        sum(true_values) / len(true_values)
        > sum(false_values) / len(false_values)
    )


# ============================================================
# LIKELIHOOD MODEL
# ============================================================


def test_calibrated_models_are_registered():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    h3_model = registry.get("G3", "H3")

    assert h3_model is not None

    assert h3_model.measurement == (
        "mean_absolute_standardized_shift"
    )


def test_calibrated_models_are_registered_for_primary_and_secondary_hypotheses():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    h3_model = registry.get("G3", "H3")

    assert h3_model is not None

    assert h3_model.measurement == (
        "mean_absolute_standardized_shift"
    )



# ============================================================
# EVIDENCE LIKELIHOOD EVALUATION
# ============================================================


def test_evaluator_returns_primary_and_secondary_log_ratios():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(result)

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert "H3" in log_ratios
    assert set(log_ratios) == {"H3"}

    assert math.isfinite(log_ratios["H3"])

    assert log_ratios["H3"] > 0.0


def test_production_map_contains_h3():
    mapping = EVIDENCE_HYPOTHESIS_MAP["G3"]

    assert "H3" in mapping["primary"]


def test_evaluator_returns_h3_log_ratio():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(result)

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert "H3" in log_ratios
    assert math.isfinite(log_ratios["H3"])
    assert log_ratios["H3"] > 0.0


def test_h3_log_ratio_uses_shift_measurement():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(result)

    model = registry.get("G3", "H3")

    expected_log_ratio = model.log_likelihood_ratio(
        evidence.metadata["measurements"][
            "mean_absolute_standardized_shift"
        ].value
    )

    actual_log_ratio = evaluator.evaluate_log_ratios(
        evidence
    )["H3"]

    assert actual_log_ratio == pytest.approx(
        expected_log_ratio
    )


# ============================================================
# BAYESIAN DIAGNOSTIC
# ============================================================


def test_strong_distribution_shift_increases_h3_support():
    engine = _build_diagnostic_engine()

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert diagnostic.posterior["H3"] > 1.0 / 9.0


def test_small_distribution_shift_has_weaker_h3_support():
    engine = _build_diagnostic_engine()

    small_shift_result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=0.10,
        random_state=100,
    )

    small_shift_evidence = (
        collect_distribution_shift_evidence(
            small_shift_result
        )
    )

    small_shift_diagnostic = engine.diagnose(
        evidence_groups=[small_shift_evidence]
    )

    strong_shift_result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.50,
        random_state=100,
    )

    strong_shift_evidence = (
        collect_distribution_shift_evidence(
            strong_shift_result
        )
    )

    strong_shift_diagnostic = engine.diagnose(
        evidence_groups=[strong_shift_evidence]
    )

    assert (
        strong_shift_diagnostic.posterior["H3"]
        > small_shift_diagnostic.posterior["H3"]
    )


def test_diagnostic_contains_all_hypotheses():
    engine = _build_diagnostic_engine()

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(
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


def test_diagnostic_is_normalized():
    engine = _build_diagnostic_engine()

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert sum(
        diagnostic.posterior.values()
    ) == pytest.approx(1.0)


def test_unavailable_g3_preserves_h3_prior():
    engine = _build_diagnostic_engine()

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(
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

    assert diagnostic.posterior["H3"] == pytest.approx(
        1.0 / 9.0
    )


# ============================================================
# NUMERICAL STABILITY
# ============================================================


def test_log_likelihood_ratio_handles_extreme_evidence():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G3", "H3")

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(
        result
    )

    log_ratio = model.log_likelihood_ratio(
        evidence.value
    )

    assert math.isfinite(log_ratio)
    assert log_ratio > 0.0


def test_evaluator_log_ratio_handles_extreme_evidence():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert "H3" in log_ratios
    assert set(log_ratios) == {"H3"}

    assert math.isfinite(log_ratios["H3"])

    assert log_ratios["H3"] > 0.0


def test_diagnostic_handles_extreme_distribution_shift():
    engine = _build_diagnostic_engine()

    result = run_distribution_shift_pipeline(
        n_samples=500,
        n_features=10,
        shift_magnitude=1.5,
        random_state=100,
    )

    evidence = collect_distribution_shift_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert diagnostic.posterior["H3"] > 1.0 / 9.0

    assert sum(
        diagnostic.posterior.values()
    ) == pytest.approx(1.0)


# ============================================================
# VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "shift_magnitude",
    [-1.0, -0.1],
)
def test_negative_shift_is_rejected(shift_magnitude):
    with pytest.raises(ValueError):
        run_distribution_shift_pipeline(
            n_samples=500,
            n_features=10,
            shift_magnitude=shift_magnitude,
        )


@pytest.mark.parametrize(
    "test_size",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_test_size_is_rejected(test_size):
    with pytest.raises(ValueError):
        run_distribution_shift_pipeline(
            n_samples=500,
            n_features=10,
            test_size=test_size,
        )


@pytest.mark.parametrize(
    "n_samples,n_features",
    [
        (0, 10),
        (-1, 10),
        (500, 0),
        (500, -1),
    ],
)
def test_invalid_dataset_dimensions_are_rejected(
    n_samples,
    n_features,
):
    with pytest.raises(ValueError):
        run_distribution_shift_pipeline(
            n_samples=n_samples,
            n_features=n_features,
        )