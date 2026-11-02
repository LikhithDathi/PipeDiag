import math

import pytest

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)


def test_default_measurement_is_supported():
    observation = CalibrationObservation(
        evidence_group="G1",
        hypothesis="H1",
        hypothesis_true=True,
        value=0.8,
    )

    assert observation.measurement == "default"


def test_named_measurement_is_stored():
    observation = CalibrationObservation(
        evidence_group="G2",
        hypothesis="H2",
        hypothesis_true=True,
        value=0.25,
        measurement="generalization_gap",
    )

    assert observation.measurement == (
        "generalization_gap"
    )


def test_training_accuracy_measurement_is_supported():
    observation = CalibrationObservation(
        evidence_group="G2",
        hypothesis="H7",
        hypothesis_true=True,
        value=0.70,
        measurement="training_accuracy",
    )

    assert observation.measurement == (
        "training_accuracy"
    )


def test_empty_measurement_is_rejected():
    with pytest.raises(ValueError):
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=True,
            value=0.70,
            measurement="",
        )


def test_measurement_is_immutable():
    observation = CalibrationObservation(
        evidence_group="G2",
        hypothesis="H7",
        hypothesis_true=True,
        value=0.70,
        measurement="training_accuracy",
    )

    with pytest.raises(Exception):
        observation.measurement = "other"


def test_non_finite_value_is_still_rejected():
    with pytest.raises(ValueError):
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=True,
            value=math.nan,
            measurement="training_accuracy",
        )