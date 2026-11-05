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

from experiments.feature_problem_calibration import (
    collect_feature_problem_calibration_observations,
)
from experiments.feature_problem_evidence import (
    collect_feature_problem_evidence,
)
from experiments.feature_problem_pipeline import (
    FeatureProblemPipelineResult,
    run_feature_problem_pipeline,
)


def _build_diagnostic_engine():
    observations = (
        collect_feature_problem_calibration_observations(
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
    result = run_feature_problem_pipeline()

    assert isinstance(
        result,
        FeatureProblemPipelineResult,
    )


def test_output_shapes_are_consistent():
    result = run_feature_problem_pipeline(
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
    result = run_feature_problem_pipeline()

    assert set(np.unique(result.y_train)) == {0, 1}
    assert set(np.unique(result.y_test)) == {0, 1}


def test_accuracy_is_valid():
    result = run_feature_problem_pipeline()

    assert 0.0 <= result.accuracy <= 1.0


def test_feature_count_is_correct():
    result = run_feature_problem_pipeline(
        n_features=12,
    )

    assert result.n_features == 12
    assert result.X_train.shape[1] == 12


def test_constant_feature_count_is_valid():
    result = run_feature_problem_pipeline(
        n_features=10,
        constant_feature_ratio=0.50,
    )

    assert result.constant_feature_count == 5


def test_constant_feature_ratio_is_valid():
    result = run_feature_problem_pipeline(
        n_features=10,
        constant_feature_ratio=0.50,
    )

    assert result.constant_feature_ratio == pytest.approx(
        0.50
    )


def test_no_injected_constant_features_gives_zero_ratio():
    result = run_feature_problem_pipeline(
        n_features=10,
        constant_feature_ratio=0.0,
    )

    assert result.constant_feature_count == 0
    assert result.constant_feature_ratio == pytest.approx(
        0.0
    )


def test_more_constant_features_increases_observed_ratio():
    low = run_feature_problem_pipeline(
        n_features=10,
        constant_feature_ratio=0.20,
        random_state=42,
    )

    high = run_feature_problem_pipeline(
        n_features=10,
        constant_feature_ratio=0.70,
        random_state=42,
    )

    assert (
        high.constant_feature_ratio
        > low.constant_feature_ratio
    )


def test_constant_features_are_actually_constant():
    result = run_feature_problem_pipeline(
        n_features=10,
        constant_feature_ratio=0.50,
    )

    for feature_index in range(5, 10):
        assert np.allclose(
            result.X_train[:, feature_index],
            0.0,
        )

        assert np.allclose(
            result.X_test[:, feature_index],
            0.0,
        )


def test_result_is_reproducible():
    result_a = run_feature_problem_pipeline(
        n_samples=500,
        n_features=10,
        constant_feature_ratio=0.60,
        random_state=42,
    )

    result_b = run_feature_problem_pipeline(
        n_samples=500,
        n_features=10,
        constant_feature_ratio=0.60,
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

    assert result_a.accuracy == pytest.approx(
        result_b.accuracy
    )

    assert (
        result_a.constant_feature_ratio
        == pytest.approx(
            result_b.constant_feature_ratio
        )
    )


# ============================================================
# EVIDENCE
# ============================================================


def test_evidence_uses_g7():
    result = run_feature_problem_pipeline(
        constant_feature_ratio=0.60,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    assert evidence.name == "G7"
    assert evidence.status == "observed"


def test_evidence_value_matches_feature_ratio():
    result = run_feature_problem_pipeline(
        constant_feature_ratio=0.60,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    assert evidence.value == pytest.approx(
        result.constant_feature_ratio
    )


def test_evidence_contains_correct_measurement():
    result = run_feature_problem_pipeline(
        constant_feature_ratio=0.60,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    measurements = evidence.metadata["measurements"]

    assert (
        "constant_feature_ratio"
        in measurements
    )

    assert measurements[
        "constant_feature_ratio"
    ].value == pytest.approx(
        result.constant_feature_ratio
    )


def test_evidence_contains_feature_metadata():
    result = run_feature_problem_pipeline(
        n_features=10,
        constant_feature_ratio=0.60,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    assert (
        evidence.metadata[
            "constant_feature_count"
        ]
        == result.constant_feature_count
    )

    assert (
        evidence.metadata["n_features"]
        == result.n_features
    )


# ============================================================
# CALIBRATION
# ============================================================


def test_calibration_returns_observations():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    assert observations


def test_calibration_observation_types_are_correct():
    observations = (
        collect_feature_problem_calibration_observations()
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
        collect_feature_problem_calibration_observations()
    )

    assert any(
        observation.hypothesis_true
        for observation in observations
    )

    assert any(
        not observation.hypothesis_true
        for observation in observations
    )


def test_calibration_uses_g7_h8():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    h8_observations = [
        observation
        for observation in observations
        if observation.hypothesis == "H8"
    ]

    assert h8_observations

    assert all(
        observation.evidence_group == "G7"
        for observation in h8_observations
    )


def test_g7_calibration_does_not_support_h7():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    assert all(
        observation.hypothesis == "H8"
        for observation in observations
    )


def test_calibration_uses_correct_measurement():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    assert all(
        observation.measurement
        == "constant_feature_ratio"
        for observation in observations
    )


def test_faulty_scenarios_have_higher_feature_problem_measurement():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    h8_observations = [
        observation
        for observation in observations
        if observation.hypothesis == "H8"
    ]

    true_values = [
        observation.value
        for observation in h8_observations
        if observation.hypothesis_true
    ]

    false_values = [
        observation.value
        for observation in h8_observations
        if not observation.hypothesis_true
    ]

    assert (
        sum(true_values) / len(true_values)
        > sum(false_values) / len(false_values)
    )


def test_calibration_values_are_finite():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    assert all(
        math.isfinite(observation.value)
        for observation in observations
    )


# ============================================================
# EVIDENCE MAPPING
# ============================================================


def test_g7_primary_mapping_contains_h8():
    assert "H8" in EVIDENCE_HYPOTHESIS_MAP[
        "G7"
    ]["primary"]


def test_g7_has_no_secondary_hypotheses():
    assert EVIDENCE_HYPOTHESIS_MAP["G7"]["secondary"] == set()


# ============================================================
# LIKELIHOOD CALIBRATION
# ============================================================


def test_g7_h8_likelihood_model_is_registered():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G7", "H8")

    assert model is not None

    assert model.measurement == (
        "constant_feature_ratio"
    )





def test_h8_model_separates_true_and_false_samples():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G7", "H8")

    assert model.mean_true > model.mean_false




# ============================================================
# EVIDENCE LIKELIHOOD
# ============================================================


def test_evaluator_returns_h8_log_ratio():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_feature_problem_pipeline(
        n_samples=500,
        n_features=10,
        constant_feature_ratio=0.70,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert "H8" in log_ratios
    assert set(log_ratios) == {"H8"}

    assert math.isfinite(log_ratios["H8"])


def test_strong_feature_problem_supports_h8():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_feature_problem_pipeline(
        n_samples=500,
        n_features=10,
        constant_feature_ratio=0.70,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert log_ratios["H8"] > 0.0


def test_normal_features_have_weaker_h8_support():
    observations = (
        collect_feature_problem_calibration_observations()
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    result = run_feature_problem_pipeline(
        n_samples=500,
        n_features=10,
        constant_feature_ratio=0.0,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    log_ratios = evaluator.evaluate_log_ratios(
        evidence
    )

    assert (
        log_ratios["H8"] < 0.0
    )


# ============================================================
# BAYESIAN DIAGNOSTIC
# ============================================================


def test_feature_problem_increases_h8_support():
    engine = _build_diagnostic_engine()

    result = run_feature_problem_pipeline(
        n_samples=500,
        n_features=10,
        constant_feature_ratio=0.70,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert (
        diagnostic.posterior["H8"]
        > 1.0 / 9.0
    )


def test_normal_features_do_not_strongly_support_h8():
    engine = _build_diagnostic_engine()

    result = run_feature_problem_pipeline(
        n_samples=500,
        n_features=10,
        constant_feature_ratio=0.0,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert (
        diagnostic.posterior["H8"]
        < 1.0 / 9.0
    )




def test_diagnostic_contains_all_hypotheses():
    engine = _build_diagnostic_engine()

    result = run_feature_problem_pipeline(
        constant_feature_ratio=0.70,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
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

    result = run_feature_problem_pipeline(
        constant_feature_ratio=0.70,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
        result
    )

    diagnostic = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert sum(
        diagnostic.posterior.values()
    ) == pytest.approx(1.0)


def test_unavailable_g7_preserves_prior():
    engine = _build_diagnostic_engine()

    result = run_feature_problem_pipeline(
        constant_feature_ratio=0.70,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
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

    assert diagnostic.posterior["H8"] == pytest.approx(
        1.0 / 9.0
    )




# ============================================================
# NUMERICAL STABILITY
# ============================================================


def test_feature_log_ratios_are_finite():
    engine = _build_diagnostic_engine()

    result = run_feature_problem_pipeline(
        constant_feature_ratio=0.90,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
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


def test_feature_diagnostic_is_numerically_stable():
    engine = _build_diagnostic_engine()

    result = run_feature_problem_pipeline(
        constant_feature_ratio=0.90,
        random_state=100,
    )

    evidence = collect_feature_problem_evidence(
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
        run_feature_problem_pipeline(
            n_samples=n_samples,
            n_features=n_features,
        )


@pytest.mark.parametrize(
    "ratio",
    [-0.1, -1.0, 1.0, 1.1],
)
def test_invalid_constant_feature_ratio_is_rejected(
    ratio,
):
    with pytest.raises(ValueError):
        run_feature_problem_pipeline(
            constant_feature_ratio=ratio,
        )


def test_empty_random_states_are_rejected():
    with pytest.raises(ValueError):
        collect_feature_problem_calibration_observations(
            random_states=(),
        )


def test_normal_ratio_length_must_match_random_states():
    with pytest.raises(ValueError):
        collect_feature_problem_calibration_observations(
            random_states=(1, 2, 3),
            normal_constant_feature_ratios=(
                0.0,
                0.1,
            ),
            faulty_constant_feature_ratios=(
                0.5,
                0.6,
                0.7,
            ),
        )


def test_faulty_ratio_length_must_match_random_states():
    with pytest.raises(ValueError):
        collect_feature_problem_calibration_observations(
            random_states=(1, 2, 3),
            normal_constant_feature_ratios=(
                0.0,
                0.1,
                0.0,
            ),
            faulty_constant_feature_ratios=(
                0.5,
                0.6,
            ),
        )


def test_invalid_normal_calibration_ratio_is_rejected():
    with pytest.raises(ValueError):
        collect_feature_problem_calibration_observations(
            normal_constant_feature_ratios=(
                0.0,
                0.1,
                1.0,
                0.0,
                0.1,
            ),
        )


def test_invalid_faulty_calibration_ratio_is_rejected():
    with pytest.raises(ValueError):
        collect_feature_problem_calibration_observations(
            faulty_constant_feature_ratios=(
                0.5,
                0.6,
                1.0,
                0.8,
                0.9,
            ),
        )