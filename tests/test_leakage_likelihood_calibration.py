import math

import pytest

from experiments.leakage_calibration import (
    collect_leakage_calibration_observations,
)
from pipediag.diagnostics.calibration import CalibrationObservation
from pipediag.diagnostics.calibration_estimator import CalibrationEstimator


def _collect_observations():
    return collect_leakage_calibration_observations(
        n_samples=500,
        n_features=10,
        random_states=(1, 2, 3, 4, 5, 6),
        leakage_strengths=(
            0.80,
            0.84,
            0.88,
            0.92,
            0.96,
            0.90,
        ),
    )


def test_estimator_produces_g1_h1_parameters():
    observations = _collect_observations()

    estimator = CalibrationEstimator()

    parameters = estimator.fit(observations)

    assert math.isfinite(parameters.mean_true)
    assert math.isfinite(parameters.std_true)
    assert math.isfinite(parameters.mean_false)
    assert math.isfinite(parameters.std_false)


def test_estimated_standard_deviations_are_positive():
    observations = _collect_observations()

    estimator = CalibrationEstimator()

    parameters = estimator.fit(observations)

    assert parameters.std_true > 0.0
    assert parameters.std_false > 0.0


def test_leakage_mean_is_higher_than_clean_mean():
    observations = _collect_observations()

    estimator = CalibrationEstimator()

    parameters = estimator.fit(observations)

    assert parameters.mean_true > parameters.mean_false


def test_parameters_are_based_on_observed_calibration_data():
    observations = _collect_observations()

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

    estimator = CalibrationEstimator()

    parameters = estimator.fit(observations)

    expected_true_mean = sum(true_values) / len(true_values)
    expected_false_mean = sum(false_values) / len(false_values)

    assert parameters.mean_true == pytest.approx(
        expected_true_mean
    )

    assert parameters.mean_false == pytest.approx(
        expected_false_mean
    )


def test_estimator_rejects_mixed_evidence_groups():
    observations = _collect_observations()

    observations.append(
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.5,
        )
    )

    estimator = CalibrationEstimator()

    with pytest.raises(ValueError):
        estimator.fit(observations)


def test_estimator_rejects_mixed_hypotheses():
    observations = _collect_observations()

    observations.append(
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H2",
            hypothesis_true=True,
            value=0.5,
        )
    )

    estimator = CalibrationEstimator()

    with pytest.raises(ValueError):
        estimator.fit(observations)