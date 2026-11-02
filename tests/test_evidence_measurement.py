import math

import pytest

from pipediag.diagnostics.evidence_measurement import (
    EvidenceMeasurement,
)


def test_measurement_stores_name_and_value():
    measurement = EvidenceMeasurement(
        name="training_accuracy",
        value=0.85,
    )

    assert measurement.name == "training_accuracy"
    assert measurement.value == pytest.approx(0.85)


def test_empty_name_is_rejected():
    with pytest.raises(ValueError):
        EvidenceMeasurement(
            name="",
            value=0.85,
        )


def test_non_numeric_value_is_rejected():
    with pytest.raises(ValueError):
        EvidenceMeasurement(
            name="training_accuracy",
            value="0.85",
        )


def test_nan_is_rejected():
    with pytest.raises(ValueError):
        EvidenceMeasurement(
            name="training_accuracy",
            value=math.nan,
        )


def test_infinity_is_rejected():
    with pytest.raises(ValueError):
        EvidenceMeasurement(
            name="training_accuracy",
            value=math.inf,
        )


def test_integer_values_are_supported():
    measurement = EvidenceMeasurement(
        name="sample_count",
        value=100,
    )

    assert measurement.value == 100


def test_measurement_is_immutable():
    measurement = EvidenceMeasurement(
        name="training_accuracy",
        value=0.85,
    )

    with pytest.raises(Exception):
        measurement.value = 0.9