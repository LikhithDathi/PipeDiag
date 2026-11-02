import pytest

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)


def test_observation_stores_required_information():
    observation = CalibrationObservation(
        evidence_group="G1",
        hypothesis="H1",
        hypothesis_true=True,
        value=0.82,
    )

    assert observation.evidence_group == "G1"
    assert observation.hypothesis == "H1"
    assert observation.hypothesis_true is True
    assert observation.value == 0.82


def test_observation_accepts_false_hypothesis_state():
    observation = CalibrationObservation(
        evidence_group="G1",
        hypothesis="H1",
        hypothesis_true=False,
        value=0.21,
    )

    assert observation.hypothesis_true is False


def test_observation_rejects_empty_evidence_group():
    with pytest.raises(ValueError):
        CalibrationObservation(
            evidence_group="",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.82,
        )


def test_observation_rejects_empty_hypothesis():
    with pytest.raises(ValueError):
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="",
            hypothesis_true=True,
            value=0.82,
        )


def test_observation_rejects_non_finite_value():
    with pytest.raises(ValueError):
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=float("nan"),
        )


def test_observation_rejects_infinite_value():
    with pytest.raises(ValueError):
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=float("inf"),
        )


def test_observation_accepts_zero():
    observation = CalibrationObservation(
        evidence_group="G1",
        hypothesis="H1",
        hypothesis_true=True,
        value=0.0,
    )

    assert observation.value == 0.0


def test_observation_accepts_negative_value():
    observation = CalibrationObservation(
        evidence_group="G7",
        hypothesis="H8",
        hypothesis_true=True,
        value=-2.5,
    )

    assert observation.value == -2.5


def test_observation_is_immutable():
    observation = CalibrationObservation(
        evidence_group="G1",
        hypothesis="H1",
        hypothesis_true=True,
        value=0.82,
    )

    with pytest.raises(AttributeError):
        observation.value = 0.5


def test_observation_requires_boolean_hypothesis_state():
    with pytest.raises(ValueError):
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=1,
            value=0.82,
        )