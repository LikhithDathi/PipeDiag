import pytest

from pipediag.diagnostics.calibration import CalibrationObservation
from pipediag.diagnostics.explanation_calibration import (
    EvidenceInterpretationCalibrator,
    InterpretationBoundary,
)


def _observations(
    *,
    evidence_group="G3",
    hypothesis="H3",
    measurement="mean_absolute_standardized_shift",
):
    return [
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=False,
            value=0.05,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=False,
            value=0.08,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=False,
            value=0.10,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=False,
            value=0.12,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=False,
            value=0.15,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=True,
            value=0.55,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=True,
            value=0.65,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=True,
            value=0.75,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=True,
            value=0.85,
            measurement=measurement,
        ),
        CalibrationObservation(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            hypothesis_true=True,
            value=0.95,
            measurement=measurement,
        ),
    ]


def test_boundary_validates_direction():
    boundary = InterpretationBoundary(
        evidence_group="G3",
        hypothesis="H3",
        measurement="shift",
        normal_reference=0.10,
        observable_boundary=0.15,
        substantial_boundary=0.65,
        direction=1,
    )

    assert boundary.classify(0.10) == "limited"
    assert boundary.classify(0.20) == "observable"
    assert boundary.classify(0.70) == "substantial"


def test_decreasing_boundary_classifies_correctly():
    boundary = InterpretationBoundary(
        evidence_group="G4",
        hypothesis="H4",
        measurement="minority_class_proportion",
        normal_reference=0.40,
        observable_boundary=0.25,
        substantial_boundary=0.15,
        direction=-1,
    )

    assert boundary.classify(0.40) == "limited"
    assert boundary.classify(0.20) == "observable"
    assert boundary.classify(0.10) == "substantial"


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="direction"):
        InterpretationBoundary(
            evidence_group="G3",
            hypothesis="H3",
            measurement="shift",
            normal_reference=0.10,
            observable_boundary=0.20,
            substantial_boundary=0.30,
            direction=0,
        )


def test_invalid_boundary_order_is_rejected():
    with pytest.raises(ValueError, match="ordered"):
        InterpretationBoundary(
            evidence_group="G3",
            hypothesis="H3",
            measurement="shift",
            normal_reference=0.10,
            observable_boundary=0.30,
            substantial_boundary=0.20,
            direction=1,
        )


def test_calibrator_requires_observations():
    calibrator = EvidenceInterpretationCalibrator()

    with pytest.raises(ValueError, match="cannot be empty"):
        calibrator.fit([])


def test_calibrator_learns_increasing_evidence():
    calibrator = EvidenceInterpretationCalibrator()
    calibrator.fit(_observations())

    assert calibrator.contains(
        "G3",
        "H3",
        "mean_absolute_standardized_shift",
    )

    boundary = calibrator.get(
        "G3",
        "H3",
        "mean_absolute_standardized_shift",
    )

    assert boundary.direction == 1
    assert boundary.normal_reference == pytest.approx(0.10)
    assert boundary.observable_boundary > boundary.normal_reference
    assert boundary.substantial_boundary >= boundary.observable_boundary


def test_calibrator_classifies_increasing_evidence():
    calibrator = EvidenceInterpretationCalibrator()
    calibrator.fit(_observations())

    assert (
        calibrator.classify(
            evidence_group="G3",
            hypothesis="H3",
            measurement="mean_absolute_standardized_shift",
            value=0.10,
        )
        == "limited"
    )

    assert (
        calibrator.classify(
            evidence_group="G3",
            hypothesis="H3",
            measurement="mean_absolute_standardized_shift",
            value=0.40,
        )
        == "observable"
    )

    assert (
        calibrator.classify(
            evidence_group="G3",
            hypothesis="H3",
            measurement="mean_absolute_standardized_shift",
            value=0.80,
        )
        == "substantial"
    )


def test_calibrator_rejects_insufficient_true_examples():
    observations = _observations()[:5]

    observations.append(
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=True,
            value=0.80,
            measurement="mean_absolute_standardized_shift",
        )
    )

    calibrator = EvidenceInterpretationCalibrator()

    with pytest.raises(ValueError, match="fault calibration observations"):
        calibrator.fit(observations)


def test_calibrator_rejects_insufficient_false_examples():
    observations = [
        observation
        for observation in _observations()
        if observation.hypothesis_true
    ]

    observations.append(
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=False,
            value=0.10,
            measurement="mean_absolute_standardized_shift",
        )
    )

    calibrator = EvidenceInterpretationCalibrator()

    with pytest.raises(
        ValueError,
        match="non-fault calibration observations",
    ):
        calibrator.fit(observations)


def test_missing_boundary_raises_key_error():
    calibrator = EvidenceInterpretationCalibrator()

    with pytest.raises(KeyError, match="No interpretation boundary"):
        calibrator.get(
            "G3",
            "H3",
            "mean_absolute_standardized_shift",
        )


def test_boundaries_returns_copy():
    calibrator = EvidenceInterpretationCalibrator()
    calibrator.fit(_observations())

    boundaries = calibrator.boundaries()

    assert len(boundaries) == 1

    boundaries.clear()

    assert calibrator.contains(
        "G3",
        "H3",
        "mean_absolute_standardized_shift",
    )


def test_invalid_normal_quantile_is_rejected():
    with pytest.raises(ValueError, match="normal_quantile"):
        EvidenceInterpretationCalibrator(
            normal_quantile=0.5
        )


def test_non_finite_value_is_rejected():
    boundary = InterpretationBoundary(
        evidence_group="G3",
        hypothesis="H3",
        measurement="shift",
        normal_reference=0.10,
        observable_boundary=0.20,
        substantial_boundary=0.30,
        direction=1,
    )

    with pytest.raises(ValueError, match="finite"):
        boundary.classify(float("inf"))