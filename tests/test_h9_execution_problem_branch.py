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

from experiments.execution_problem_calibration import (
    _smoothed_failure_rate,
    collect_execution_problem_calibration_observations,
)
from experiments.execution_problem_evidence import (
    collect_execution_problem_evidence,
)
from experiments.execution_problem_pipeline import (
    ExecutionProblemPipelineResult,
    run_execution_problem_pipeline,
)


def _build_diagnostic_engine():
    observations = (
        collect_execution_problem_calibration_observations(
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
# PIPELINE
# ============================================================


def test_returns_expected_result_type():
    result = run_execution_problem_pipeline()

    assert isinstance(
        result,
        ExecutionProblemPipelineResult,
    )


def test_output_shapes_are_consistent():
    result = run_execution_problem_pipeline(
        n_samples=500,
        n_features=10,
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
    result = run_execution_problem_pipeline()

    assert set(np.unique(result.y_train)) == {
        0,
        1,
    }

    assert set(np.unique(result.y_test)) == {
        0,
        1,
    }


def test_accuracy_is_valid():
    result = run_execution_problem_pipeline()

    assert 0.0 <= result.accuracy <= 1.0


def test_total_execution_events_are_positive():
    result = run_execution_problem_pipeline()

    assert result.total_execution_events > 0


def test_failed_execution_events_are_valid():
    result = run_execution_problem_pipeline()

    assert (
        0
        <= result.failed_execution_events
        <= result.total_execution_events
    )


def test_clean_pipeline_has_no_failed_events():
    result = run_execution_problem_pipeline(
        inject_failure=False,
    )

    assert result.failed_execution_events == 0

    assert result.execution_failure_rate == pytest.approx(
        0.0
    )


def test_faulty_pipeline_records_execution_failure():
    result = run_execution_problem_pipeline(
        inject_failure=True,
    )

    assert result.failed_execution_events > 0


def test_faulty_pipeline_has_positive_failure_rate():
    result = run_execution_problem_pipeline(
        inject_failure=True,
    )

    assert result.execution_failure_rate > 0.0


def test_failure_rate_is_correctly_calculated():
    result = run_execution_problem_pipeline(
        inject_failure=True,
    )

    expected = (
        result.failed_execution_events
        / result.total_execution_events
    )

    assert result.execution_failure_rate == pytest.approx(
        expected
    )


def test_failure_rate_is_finite():
    result = run_execution_problem_pipeline(
        inject_failure=True,
    )

    assert math.isfinite(
        result.execution_failure_rate
    )


def test_fault_injection_flag_is_recorded():
    clean = run_execution_problem_pipeline(
        inject_failure=False,
    )

    faulty = run_execution_problem_pipeline(
        inject_failure=True,
    )

    assert clean.fault_injected is False
    assert faulty.fault_injected is True


def test_same_random_state_is_reproducible():
    result_a = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=42,
    )

    result_b = run_execution_problem_pipeline(
        inject_failure=True,
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

    assert result_a.predictions.tolist() == (
        result_b.predictions.tolist()
    )

    assert result_a.accuracy == pytest.approx(
        result_b.accuracy
    )

    assert (
        result_a.execution_failure_rate
        == pytest.approx(
            result_b.execution_failure_rate
        )
    )


def test_more_execution_failures_increase_failure_rate():
    low = run_execution_problem_pipeline(
        inject_failure=True,
        failure_event_count=1,
    )

    high = run_execution_problem_pipeline(
        inject_failure=True,
        failure_event_count=2,
    )

    assert (
        high.execution_failure_rate
        > low.execution_failure_rate
    )


# ============================================================
# EVIDENCE
# ============================================================


def test_evidence_uses_g8():
    result = run_execution_problem_pipeline(
        inject_failure=True,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    assert evidence.name == "G8"
    assert evidence.status == "observed"


def test_evidence_value_matches_failure_rate():
    result = run_execution_problem_pipeline(
        inject_failure=True,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    assert evidence.value == pytest.approx(
        result.execution_failure_rate
    )


def test_evidence_contains_correct_measurement():
    result = run_execution_problem_pipeline(
        inject_failure=True,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    measurements = evidence.metadata[
        "measurements"
    ]

    assert (
        "execution_failure_rate"
        in measurements
    )

    assert measurements[
        "execution_failure_rate"
    ].value == pytest.approx(
        result.execution_failure_rate
    )


def test_evidence_contains_execution_metadata():
    result = run_execution_problem_pipeline(
        inject_failure=True,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    assert (
        evidence.metadata[
            "total_execution_events"
        ]
        == result.total_execution_events
    )

    assert (
        evidence.metadata[
            "failed_execution_events"
        ]
        == result.failed_execution_events
    )


# ============================================================
# CALIBRATION
# ============================================================


def test_calibration_returns_observations():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    assert observations


def test_calibration_returns_correct_types():
    observations = (
        collect_execution_problem_calibration_observations()
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
        collect_execution_problem_calibration_observations()
    )

    assert any(
        observation.hypothesis_true
        for observation in observations
    )

    assert any(
        not observation.hypothesis_true
        for observation in observations
    )


def test_calibration_uses_g8_h9():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    assert all(
        observation.evidence_group == "G8"
        for observation in observations
    )

    assert all(
        observation.hypothesis == "H9"
        for observation in observations
    )


def test_calibration_uses_correct_measurement():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    assert all(
        observation.measurement
        == "execution_failure_rate"
        for observation in observations
    )


def test_faulty_scenarios_have_higher_failure_rate():
    observations = (
        collect_execution_problem_calibration_observations()
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


def test_calibration_values_are_finite():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    assert all(
        math.isfinite(observation.value)
        for observation in observations
    )


def test_calibration_values_have_variance_for_both_classes():
    observations = (
        collect_execution_problem_calibration_observations()
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

    assert np.var(true_values) > 0.0
    assert np.var(false_values) > 0.0


def test_clean_calibration_values_are_smoothed():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    false_values = [
        observation.value
        for observation in observations
        if not observation.hypothesis_true
    ]

    assert all(
        value > 0.0
        for value in false_values
    )


def test_calibration_preserves_fault_separation():
    observations = (
        collect_execution_problem_calibration_observations()
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
        np.mean(true_values)
        > np.mean(false_values)
    )


def test_smoothed_failure_rate_handles_zero_failures():
    value = _smoothed_failure_rate(
        failed_execution_events=0,
        total_execution_events=3,
    )

    assert value == pytest.approx(0.125)


# ============================================================
# EVIDENCE MAPPING
# ============================================================


def test_g8_primary_mapping_contains_h9():
    assert "H9" in EVIDENCE_HYPOTHESIS_MAP[
        "G8"
    ]["primary"]


def test_g8_has_no_secondary_hypotheses():
    assert not EVIDENCE_HYPOTHESIS_MAP[
        "G8"
    ]["secondary"]


def test_failure_event_counts_length_must_match_random_states():
    with pytest.raises(ValueError):
        collect_execution_problem_calibration_observations(
            random_states=(1, 2, 3),
            failure_event_counts=(1, 2),
        )


def test_failure_event_counts_must_be_positive():
    with pytest.raises(ValueError):
        collect_execution_problem_calibration_observations(
            failure_event_counts=(1, 0, 2, 1, 1),
        )


# ============================================================
# LIKELIHOOD CALIBRATION
# ============================================================


def test_g8_h9_likelihood_model_is_registered():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G8", "H9")

    assert model is not None
    assert model.measurement == (
        "execution_failure_rate"
    )


def test_h9_model_separates_true_and_false_samples():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G8", "H9")

    assert model.mean_true > model.mean_false


# ============================================================
# EVIDENCE LIKELIHOOD
# ============================================================


def test_evaluator_returns_h9_log_ratio():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert "H9" in log_ratios
    assert math.isfinite(log_ratios["H9"])


def test_execution_failure_supports_h9():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert log_ratios["H9"] > 0.0


def test_clean_execution_has_weaker_h9_support():
    observations = (
        collect_execution_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_execution_problem_pipeline(
        inject_failure=False,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert log_ratios["H9"] < 0.0


# ============================================================
# BAYESIAN DIAGNOSTIC
# ============================================================


def test_execution_failure_increases_h9_support():
    engine = _build_diagnostic_engine()

    result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert (
        diagnostic.posterior["H9"]
        > 1.0 / 9.0
    )


def test_clean_execution_has_weaker_h9_support_bayesian():
    engine = _build_diagnostic_engine()

    result = run_execution_problem_pipeline(
        inject_failure=False,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert (
        diagnostic.posterior["H9"]
        < 1.0 / 9.0
    )


def test_faulty_execution_has_more_h9_support_than_clean_execution():
    engine = _build_diagnostic_engine()

    clean_result = run_execution_problem_pipeline(
        inject_failure=False,
        random_state=100,
    )

    clean_evidence = (
        collect_execution_problem_evidence(
            clean_result
        )
    )

    clean_diagnostic = engine.diagnose(
        evidence_groups=[clean_evidence]
    )

    faulty_result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    faulty_evidence = (
        collect_execution_problem_evidence(
            faulty_result
        )
    )

    faulty_diagnostic = engine.diagnose(
        evidence_groups=[faulty_evidence]
    )

    assert (
        faulty_diagnostic.posterior["H9"]
        > clean_diagnostic.posterior["H9"]
    )


def test_diagnostic_contains_all_hypotheses():
    engine = _build_diagnostic_engine()

    result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
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

    result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert sum(
        diagnostic.posterior.values()
    ) == pytest.approx(1.0)


def test_unavailable_g8_preserves_h9_prior():
    engine = _build_diagnostic_engine()

    result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
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

    assert diagnostic.posterior["H9"] == pytest.approx(
        1.0 / 9.0
    )


# ============================================================
# NUMERICAL STABILITY
# ============================================================


def test_h9_log_ratio_is_finite():
    engine = _build_diagnostic_engine()

    result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
        result
    )

    log_ratios = (
        engine.likelihood_evaluator
        .evaluate_log_ratios(evidence)
    )

    assert math.isfinite(
        log_ratios["H9"]
    )


def test_h9_diagnostic_is_numerically_stable():
    engine = _build_diagnostic_engine()

    result = run_execution_problem_pipeline(
        inject_failure=True,
        random_state=100,
    )

    evidence = collect_execution_problem_evidence(
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
    "n_samples,n_features",
    [
        (0, 10),
        (-1, 10),
        (500, 0),
        (500, -1),
    ],
)
def test_invalid_dimensions_are_rejected(
    n_samples,
    n_features,
):
    with pytest.raises(ValueError):
        run_execution_problem_pipeline(
            n_samples=n_samples,
            n_features=n_features,
        )