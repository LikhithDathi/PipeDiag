import math

import numpy as np
import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.diagnostic_engine import (
    DiagnosticEngine,
)
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
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

from experiments.data_quality_calibration import (
    collect_data_quality_calibration_observations,
)
from experiments.data_quality_evidence import (
    collect_data_quality_evidence,
)
from experiments.data_quality_pipeline import (
    DataQualityPipelineResult,
    run_data_quality_pipeline,
)


def _build_diagnostic_engine():
    observations = (
        collect_data_quality_calibration_observations(
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
# DATA QUALITY PIPELINE
# ============================================================


def test_returns_expected_result_type():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert isinstance(
        result,
        DataQualityPipelineResult,
    )


def test_output_shapes_are_consistent():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert result.X_train.ndim == 2
    assert result.X_test.ndim == 2

    assert result.X_train.shape[1] == 10
    assert result.X_test.shape[1] == 10

    assert result.X_train.shape[0] == len(
        result.y_train
    )

    assert result.X_test.shape[0] == len(
        result.y_test
    )


def test_target_values_are_binary():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert set(np.unique(result.y_train)) == {
        0,
        1,
    }

    assert set(np.unique(result.y_test)) == {
        0,
        1,
    }


def test_clean_pipeline_has_zero_missing_values():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.0,
        random_state=42,
    )

    assert result.missing_value_rate == pytest.approx(
        0.0
    )

    assert not np.isnan(result.X_train).any()
    assert not np.isnan(result.X_test).any()


def test_missing_values_are_injected():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    assert np.isnan(result.X_train).any()
    assert np.isnan(result.X_test).any()

    assert result.missing_value_rate > 0.0


def test_missing_value_rate_matches_requested_rate():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    assert result.missing_value_rate == pytest.approx(
        0.20,
        abs=0.01,
    )


def test_larger_missing_rate_produces_more_missing_values():
    small = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.10,
        random_state=42,
    )

    large = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=42,
    )

    assert (
        large.missing_value_rate
        > small.missing_value_rate
    )


def test_pipeline_produces_predictions():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    assert len(result.predictions) == len(
        result.y_test
    )


def test_accuracy_is_valid():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    assert 0.0 <= result.accuracy <= 1.0


def test_same_random_state_is_reproducible():
    result_a = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    result_b = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    assert np.array_equal(
        result_a.X_train,
        result_b.X_train,
        equal_nan=True,
    )

    assert np.array_equal(
        result_a.X_test,
        result_b.X_test,
        equal_nan=True,
    )

    assert np.array_equal(
        result_a.y_train,
        result_b.y_train,
    )

    assert np.array_equal(
        result_a.y_test,
        result_b.y_test,
    )

    assert np.array_equal(
        result_a.predictions,
        result_b.predictions,
    )

    assert result_a.accuracy == pytest.approx(
        result_b.accuracy
    )

    assert result_a.missing_value_rate == pytest.approx(
        result_b.missing_value_rate
    )


# ============================================================
# EVIDENCE
# ============================================================


def test_evidence_has_correct_group():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    assert evidence.name == "G5"
    assert evidence.status == "observed"


def test_evidence_value_matches_missing_rate():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    assert evidence.value == pytest.approx(
        result.missing_value_rate
    )


def test_evidence_contains_named_measurement():
    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.20,
        random_state=42,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    measurements = evidence.metadata[
        "measurements"
    ]

    assert "missing_value_rate" in measurements

    measurement = measurements[
        "missing_value_rate"
    ]

    assert measurement.value == pytest.approx(
        result.missing_value_rate
    )


# ============================================================
# CALIBRATION
# ============================================================


def test_calibration_produces_observations():
    observations = (
        collect_data_quality_calibration_observations()
    )

    assert observations


def test_calibration_returns_correct_observation_type():
    observations = (
        collect_data_quality_calibration_observations()
    )

    assert all(
        isinstance(
            observation,
            CalibrationObservation,
        )
        for observation in observations
    )


def test_calibration_contains_true_and_false_samples():
    observations = (
        collect_data_quality_calibration_observations()
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


def test_calibration_uses_g5_and_mapped_hypotheses():
    observations = (
        collect_data_quality_calibration_observations()
    )

    assert all(
        observation.evidence_group == "G5"
        for observation in observations
    )

    hypotheses = {
        observation.hypothesis
        for observation in observations
    }

    assert hypotheses == {"H5", "H7"}


def test_calibration_uses_correct_measurement():
    observations = (
        collect_data_quality_calibration_observations()
    )

    assert all(
        observation.measurement
        == "missing_value_rate"
        for observation in observations
    )


def test_faulty_samples_have_larger_missing_value_rates():
    observations = (
        collect_data_quality_calibration_observations()
    )

    h5_true_values = [
        observation.value
        for observation in observations
        if observation.hypothesis == "H5"
        and observation.hypothesis_true
    ]

    h5_false_values = [
        observation.value
        for observation in observations
        if observation.hypothesis == "H5"
        and not observation.hypothesis_true
    ]

    assert (
        sum(h5_true_values) / len(h5_true_values)
        > sum(h5_false_values) / len(h5_false_values)
    )


def test_h7_secondary_calibration_is_present():
    observations = (
        collect_data_quality_calibration_observations()
    )

    h7_observations = [
        observation
        for observation in observations
        if observation.hypothesis == "H7"
    ]

    assert h7_observations

    assert any(
        observation.hypothesis_true
        for observation in h7_observations
    )

    assert any(
        not observation.hypothesis_true
        for observation in h7_observations
    )


# ============================================================
# LIKELIHOOD MODELS
# ============================================================


def test_calibrated_h5_model_is_registered():
    observations = (
        collect_data_quality_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G5", "H5")

    assert model is not None

    assert model.measurement == (
        "missing_value_rate"
    )


def test_calibrated_h7_secondary_model_is_registered():
    observations = (
        collect_data_quality_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G5", "H7")

    assert model is not None

    assert model.measurement == (
        "missing_value_rate"
    )


def test_h5_model_separates_true_and_false_samples():
    observations = (
        collect_data_quality_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G5", "H5")

    assert model.mean_true > model.mean_false


# ============================================================
# EVIDENCE LIKELIHOOD EVALUATION
# ============================================================


def test_evaluator_returns_h5_log_ratio():
    observations = (
        collect_data_quality_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert "H5" in log_ratios
    assert set(log_ratios) == {"H5"}

    assert math.isfinite(log_ratios["H5"])


def test_strong_missingness_supports_h5():
    observations = (
        collect_data_quality_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert log_ratios["H5"] > 0.0


def test_low_missingness_has_weaker_h5_support():
    observations = (
        collect_data_quality_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    small_result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.01,
        random_state=100,
    )

    large_result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    small_evidence = collect_data_quality_evidence(
        small_result
    )

    large_evidence = collect_data_quality_evidence(
        large_result
    )

    small_log_ratio = evaluator.evaluate_log_ratios(
        small_evidence
    )["H5"]

    large_log_ratio = evaluator.evaluate_log_ratios(
        large_evidence
    )["H5"]

    assert large_log_ratio > small_log_ratio


# ============================================================
# BAYESIAN DIAGNOSTIC
# ============================================================


def test_strong_missingness_increases_h5_support():
    engine = _build_diagnostic_engine()

    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert diagnostic.posterior["H5"] > 1.0 / 9.0


def test_low_missingness_has_weaker_h5_support():
    engine = _build_diagnostic_engine()

    small_result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.01,
        random_state=100,
    )

    small_evidence = collect_data_quality_evidence(
        small_result
    )

    small_diagnostic = engine.diagnose(
        evidence_groups=[small_evidence]
    )

    strong_result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    strong_evidence = collect_data_quality_evidence(
        strong_result
    )

    strong_diagnostic = engine.diagnose(
        evidence_groups=[strong_evidence]
    )

    assert (
        strong_diagnostic.posterior["H5"]
        > small_diagnostic.posterior["H5"]
    )


def test_diagnostic_contains_all_hypotheses():
    engine = _build_diagnostic_engine()

    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    evidence = collect_data_quality_evidence(
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

    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert sum(
        diagnostic.posterior.values()
    ) == pytest.approx(1.0)


def test_unavailable_g5_preserves_h5_prior():
    engine = _build_diagnostic_engine()

    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    evidence = collect_data_quality_evidence(
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

    assert diagnostic.posterior["H5"] == pytest.approx(
        1.0 / 9.0
    )


# ============================================================
# NUMERICAL STABILITY
# ============================================================


def test_h5_log_ratios_are_finite():
    engine = _build_diagnostic_engine()

    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    log_ratios = (
        engine.likelihood_evaluator
        .evaluate_log_ratios(evidence)
    )

    assert all(
        math.isfinite(value)
        for value in log_ratios.values()
    )


def test_h5_diagnostic_remains_normalized_under_strong_evidence():
    engine = _build_diagnostic_engine()

    result = run_data_quality_pipeline(
        n_samples=500,
        n_features=10,
        missing_rate=0.30,
        random_state=100,
    )

    evidence = collect_data_quality_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert all(
        math.isfinite(value)
        for value in diagnostic.posterior.values()
    )

    assert sum(
        diagnostic.posterior.values()
    ) == pytest.approx(1.0)


# ============================================================
# VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "missing_rate",
    [-0.1, -0.01, 1.0, 1.1],
)
def test_invalid_missing_rate_is_rejected(
    missing_rate,
):
    with pytest.raises(ValueError):
        run_data_quality_pipeline(
            n_samples=500,
            n_features=10,
            missing_rate=missing_rate,
        )


@pytest.mark.parametrize(
    "test_size",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_test_size_is_rejected(test_size):
    with pytest.raises(ValueError):
        run_data_quality_pipeline(
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
        run_data_quality_pipeline(
            n_samples=n_samples,
            n_features=n_features,
        )


def test_empty_random_states_are_rejected():
    with pytest.raises(ValueError):
        collect_data_quality_calibration_observations(
            random_states=(),
        )


def test_calibration_rate_lengths_must_match():
    with pytest.raises(ValueError):
        collect_data_quality_calibration_observations(
            random_states=(1, 2, 3),
            normal_missing_rates=(
                0.0,
                0.01,
            ),
            faulty_missing_rates=(
                0.10,
                0.15,
                0.20,
            ),
        )


def test_negative_normal_calibration_rate_is_rejected():
    with pytest.raises(ValueError):
        collect_data_quality_calibration_observations(
            normal_missing_rates=(
                -0.01,
                0.01,
                0.02,
                0.00,
                0.01,
            ),
        )


def test_non_positive_faulty_calibration_rate_is_rejected():
    with pytest.raises(ValueError):
        collect_data_quality_calibration_observations(
            faulty_missing_rates=(
                0.10,
                0.15,
                0.0,
                0.25,
                0.30,
            ),
        )