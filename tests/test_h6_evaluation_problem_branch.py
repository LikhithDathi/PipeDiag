
from functools import lru_cache
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

from experiments.evaluation_problem_calibration import (
    collect_evaluation_problem_calibration_observations,
)
from experiments.evaluation_problem_evidence import (
    collect_evaluation_problem_evidence,
)
from experiments.evaluation_problem_pipeline import (
    DEFAULT_MAX_WILSON_HALF_WIDTH,
    EvaluationProblemPipelineResult,
    calculate_wilson_interval_half_width,
    run_evaluation_problem_pipeline,
)


# ============================================================
# SHARED TEST HELPERS
# ============================================================

@lru_cache(maxsize=1)
def _calibration_observations():
    return tuple(
        collect_evaluation_problem_calibration_observations()
    )


def _build_diagnostic_engine():
    registry = LikelihoodModelBuilder().build(
        _calibration_observations()
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

    evidence_inference = BayesianEvidenceInference(bayesian)

    return DiagnosticEngine(
        likelihood_evaluator=evaluator,
        bayesian_evidence_inference=evidence_inference,
    )


# ============================================================
# EVALUATION PIPELINE
# ============================================================

def test_returns_expected_result_type():
    result = run_evaluation_problem_pipeline()

    assert isinstance(result, EvaluationProblemPipelineResult)


def test_output_shapes_are_consistent():
    result = run_evaluation_problem_pipeline(
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
    result = run_evaluation_problem_pipeline()

    assert set(np.unique(result.y_train)) == {0, 1}
    assert set(np.unique(result.y_test)) == {0, 1}


def test_evaluation_sample_count_matches_test_set():
    result = run_evaluation_problem_pipeline(
        test_size=0.20,
        random_state=42,
    )

    assert result.evaluation_sample_count == len(result.y_test)


def test_accuracy_is_valid():
    result = run_evaluation_problem_pipeline()

    assert 0.0 <= result.accuracy <= 1.0


def test_wilson_interval_half_width_is_non_negative():
    result = run_evaluation_problem_pipeline()

    assert result.wilson_interval_half_width >= 0.0


def test_wilson_interval_half_width_is_finite():
    result = run_evaluation_problem_pipeline()

    assert math.isfinite(result.wilson_interval_half_width)


def test_wilson_interval_half_width_is_bounded():
    result = run_evaluation_problem_pipeline()

    assert 0.0 <= result.wilson_interval_half_width <= 1.0


def test_smaller_test_set_has_larger_evaluation_uncertainty():
    reliable = run_evaluation_problem_pipeline(
        test_size=0.20,
        random_state=42,
    )

    unreliable = run_evaluation_problem_pipeline(
        test_size=0.02,
        random_state=42,
    )

    assert (
        unreliable.wilson_interval_half_width
        > reliable.wilson_interval_half_width
    )


def test_precision_adequacy_uses_configured_threshold():
    result = run_evaluation_problem_pipeline(
        max_wilson_half_width=0.05,
    )

    assert result.evaluation_precision_adequate == (
        result.wilson_interval_half_width <= 0.05
    )


def test_custom_precision_threshold_is_respected():
    result = run_evaluation_problem_pipeline(
        max_wilson_half_width=0.01,
    )

    assert result.max_wilson_half_width == 0.01
    assert result.evaluation_precision_adequate == (
        result.wilson_interval_half_width <= 0.01
    )


def test_default_precision_threshold_is_used():
    result = run_evaluation_problem_pipeline()

    assert (
        result.max_wilson_half_width
        == DEFAULT_MAX_WILSON_HALF_WIDTH
    )


def test_wilson_calculation_matches_pipeline_result():
    result = run_evaluation_problem_pipeline()

    expected = calculate_wilson_interval_half_width(
        accuracy=result.accuracy,
        sample_count=result.evaluation_sample_count,
    )

    assert result.wilson_interval_half_width == pytest.approx(
        expected
    )


def test_same_random_state_is_reproducible():
    result_a = run_evaluation_problem_pipeline(
        test_size=0.20,
        random_state=42,
    )

    result_b = run_evaluation_problem_pipeline(
        test_size=0.20,
        random_state=42,
    )

    assert np.array_equal(result_a.X_train, result_b.X_train)
    assert np.array_equal(result_a.X_test, result_b.X_test)
    assert np.array_equal(result_a.y_train, result_b.y_train)
    assert np.array_equal(result_a.y_test, result_b.y_test)
    assert np.array_equal(result_a.predictions, result_b.predictions)

    assert result_a.accuracy == pytest.approx(result_b.accuracy)

    assert result_a.wilson_interval_half_width == pytest.approx(
        result_b.wilson_interval_half_width
    )

    assert (
        result_a.evaluation_precision_adequate
        == result_b.evaluation_precision_adequate
    )


# ============================================================
# EVIDENCE
# ============================================================

def test_evidence_has_correct_group():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    assert evidence.name == "G6"
    assert evidence.status == "observed"


def test_evidence_value_matches_wilson_half_width():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    assert evidence.value == pytest.approx(
        result.wilson_interval_half_width
    )


def test_evidence_contains_correct_metric_name():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    assert evidence.metadata["metric"] == (
        "wilson_interval_half_width"
    )


def test_evidence_contains_raw_wilson_measurement():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    measurements = evidence.metadata["measurements"]

    assert "wilson_interval_half_width" in measurements

    assert measurements[
        "wilson_interval_half_width"
    ].value == pytest.approx(
        result.wilson_interval_half_width
    )


def test_evidence_contains_log_wilson_measurement():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    measurements = evidence.metadata["measurements"]

    assert "log_wilson_interval_half_width" in measurements

    assert measurements[
        "log_wilson_interval_half_width"
    ].value == pytest.approx(
        math.log(result.wilson_interval_half_width)
    )


def test_evidence_contains_evaluation_sample_count():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    assert (
        evidence.metadata["evaluation_sample_count"]
        == result.evaluation_sample_count
    )


def test_evidence_contains_accuracy():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    assert evidence.metadata["accuracy"] == pytest.approx(
        result.accuracy
    )


def test_evidence_contains_precision_threshold():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    assert (
        evidence.metadata["max_wilson_half_width"]
        == result.max_wilson_half_width
    )


def test_evidence_contains_precision_adequacy():
    result = run_evaluation_problem_pipeline()
    evidence = collect_evaluation_problem_evidence(result)

    assert (
        evidence.metadata["evaluation_precision_adequate"]
        == result.evaluation_precision_adequate
    )


# ============================================================
# CALIBRATION
# ============================================================

def test_calibration_produces_observations():
    assert _calibration_observations()


def test_calibration_returns_correct_observation_type():
    assert all(
        isinstance(observation, CalibrationObservation)
        for observation in _calibration_observations()
    )


def test_calibration_contains_true_and_false_samples():
    observations = _calibration_observations()

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


def test_calibration_uses_g6_and_h6():
    observations = _calibration_observations()

    assert all(
        observation.evidence_group == "G6"
        for observation in observations
    )

    assert all(
        observation.hypothesis == "H6"
        for observation in observations
    )


def test_calibration_uses_log_wilson_measurement():
    observations = _calibration_observations()

    assert all(
        observation.measurement
        == "log_wilson_interval_half_width"
        for observation in observations
    )


def test_calibration_values_are_finite():
    observations = _calibration_observations()

    assert all(
        math.isfinite(observation.value)
        for observation in observations
    )


def test_calibration_values_are_log_transformed():
    observations = _calibration_observations()

    assert all(
        observation.value < 0.0
        for observation in observations
    )


def test_unreliable_evaluation_has_larger_log_wilson_value():
    observations = _calibration_observations()

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

    assert true_values
    assert false_values
    assert np.mean(true_values) > np.mean(false_values)


def test_calibration_contains_varied_measurements():
    observations = _calibration_observations()

    values = {observation.value for observation in observations}

    assert len(values) > 2


def test_calibrated_h6_model_is_registered():
    registry = LikelihoodModelBuilder().build(
        _calibration_observations()
    )

    model = registry.get("G6", "H6")

    assert model is not None
    assert model.measurement == "log_wilson_interval_half_width"


def test_h6_model_separates_true_and_false_samples():
    registry = LikelihoodModelBuilder().build(
        _calibration_observations()
    )

    model = registry.get("G6", "H6")

    assert model.mean_true > model.mean_false


# ============================================================
# LIKELIHOOD EVALUATION
# ============================================================

def test_evaluator_returns_h6_log_ratio():
    registry = LikelihoodModelBuilder().build(
        _calibration_observations()
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    evidence = collect_evaluation_problem_evidence(result)
    log_ratios = evaluator.evaluate_log_ratios(evidence)

    assert "H6" in log_ratios
    assert math.isfinite(log_ratios["H6"])


def test_strong_evaluation_uncertainty_supports_h6():
    registry = LikelihoodModelBuilder().build(
        _calibration_observations()
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    evidence = collect_evaluation_problem_evidence(result)
    log_ratios = evaluator.evaluate_log_ratios(evidence)

    assert log_ratios["H6"] > 0.0


def test_reliable_evaluation_has_weaker_h6_support():
    registry = LikelihoodModelBuilder().build(
        _calibration_observations()
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    reliable_result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.20,
        random_state=100,
    )

    unreliable_result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    reliable_evidence = collect_evaluation_problem_evidence(
        reliable_result
    )

    unreliable_evidence = collect_evaluation_problem_evidence(
        unreliable_result
    )

    reliable_log_ratio = evaluator.evaluate_log_ratios(
        reliable_evidence
    )["H6"]

    unreliable_log_ratio = evaluator.evaluate_log_ratios(
        unreliable_evidence
    )["H6"]

    assert unreliable_log_ratio > reliable_log_ratio


def test_clean_evaluation_does_not_support_h6():
    registry = LikelihoodModelBuilder().build(
        _calibration_observations()
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_evaluation_problem_pipeline(
        n_samples=5000,
        test_size=0.50,
        random_state=42,
    )

    evidence = collect_evaluation_problem_evidence(result)
    log_ratios = evaluator.evaluate_log_ratios(evidence)

    assert log_ratios["H6"] < 0.0


# ============================================================
# BAYESIAN DIAGNOSTIC
# ============================================================

def test_unreliable_evaluation_increases_h6_support():
    engine = _build_diagnostic_engine()

    result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    evidence = collect_evaluation_problem_evidence(result)
    diagnostic = engine.diagnose(evidence_groups=[evidence])

    assert diagnostic.posterior["H6"] > 1.0 / 9.0


def test_reliable_evaluation_has_weaker_h6_support():
    engine = _build_diagnostic_engine()

    reliable_result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.20,
        random_state=100,
    )

    unreliable_result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    reliable_evidence = collect_evaluation_problem_evidence(
        reliable_result
    )

    unreliable_evidence = collect_evaluation_problem_evidence(
        unreliable_result
    )

    reliable_diagnostic = engine.diagnose(
        evidence_groups=[reliable_evidence]
    )

    unreliable_diagnostic = engine.diagnose(
        evidence_groups=[unreliable_evidence]
    )

    assert (
        unreliable_diagnostic.posterior["H6"]
        > reliable_diagnostic.posterior["H6"]
    )


def test_diagnostic_contains_all_hypotheses():
    engine = _build_diagnostic_engine()

    result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    evidence = collect_evaluation_problem_evidence(result)
    diagnostic = engine.diagnose(evidence_groups=[evidence])

    expected = {
        hypothesis.id
        for hypothesis in DIAGNOSTIC_HYPOTHESES
    }

    assert set(diagnostic.posterior) == expected


def test_diagnostic_is_normalized():
    engine = _build_diagnostic_engine()

    result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    evidence = collect_evaluation_problem_evidence(result)
    diagnostic = engine.diagnose(evidence_groups=[evidence])

    assert sum(
        diagnostic.posterior.values()
    ) == pytest.approx(1.0)


def test_unavailable_g6_preserves_h6_prior():
    engine = _build_diagnostic_engine()

    result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    evidence = collect_evaluation_problem_evidence(result)

    unavailable = evidence.__class__(
        name=evidence.name,
        value=evidence.value,
        status="unavailable",
        metadata=evidence.metadata,
    )

    diagnostic = engine.diagnose(
        evidence_groups=[unavailable]
    )

    assert diagnostic.posterior["H6"] == pytest.approx(
        1.0 / 9.0
    )


# ============================================================
# NUMERICAL STABILITY
# ============================================================

def test_h6_log_ratio_is_finite():
    engine = _build_diagnostic_engine()

    result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    evidence = collect_evaluation_problem_evidence(result)

    log_ratios = (
        engine.likelihood_evaluator.evaluate_log_ratios(evidence)
    )

    assert math.isfinite(log_ratios["H6"])


def test_h6_diagnostic_is_numerically_stable():
    engine = _build_diagnostic_engine()

    result = run_evaluation_problem_pipeline(
        n_samples=500,
        test_size=0.02,
        random_state=100,
    )

    evidence = collect_evaluation_problem_evidence(result)
    diagnostic = engine.diagnose(evidence_groups=[evidence])

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
    "test_size",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_test_size_is_rejected(test_size):
    with pytest.raises(ValueError):
        run_evaluation_problem_pipeline(
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
        run_evaluation_problem_pipeline(
            n_samples=n_samples,
            n_features=n_features,
        )


@pytest.mark.parametrize(
    "threshold",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_precision_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        run_evaluation_problem_pipeline(
            max_wilson_half_width=threshold,
        )


@pytest.mark.parametrize(
    "accuracy,sample_count",
    [
        (-0.1, 100),
        (1.1, 100),
        (0.5, 0),
        (0.5, -1),
    ],
)
def test_invalid_wilson_calculation_inputs_are_rejected(
    accuracy,
    sample_count,
):
    with pytest.raises(ValueError):
        calculate_wilson_interval_half_width(
            accuracy=accuracy,
            sample_count=sample_count,
        )


def test_empty_calibration_random_states_are_rejected():
    with pytest.raises(ValueError):
        collect_evaluation_problem_calibration_observations(
            random_states=(),
        )


def test_empty_calibration_sample_sizes_are_rejected():
    with pytest.raises(ValueError):
        collect_evaluation_problem_calibration_observations(
            n_samples_values=(),
        )


def test_invalid_calibration_sample_sizes_are_rejected():
    with pytest.raises(ValueError):
        collect_evaluation_problem_calibration_observations(
            n_samples_values=(200, 0),
        )


def test_empty_calibration_test_sizes_are_rejected():
    with pytest.raises(ValueError):
        collect_evaluation_problem_calibration_observations(
            test_sizes=(),
        )


def test_invalid_calibration_test_sizes_are_rejected():
    with pytest.raises(ValueError):
        collect_evaluation_problem_calibration_observations(
            test_sizes=(0.20, 1.0),
        )


def test_invalid_calibration_precision_threshold_is_rejected():
    with pytest.raises(ValueError):
        collect_evaluation_problem_calibration_observations(
            max_wilson_half_width=0.0,
        )
