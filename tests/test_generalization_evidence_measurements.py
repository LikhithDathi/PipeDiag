import pytest

from pipediag.diagnostics.evidence_measurement import (
    EvidenceMeasurement,
)
from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import (
    run_overfitting_pipeline,
)


def _build_evidence():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    return result, collect_generalization_evidence(result)


def test_g2_contains_measurements():
    _, evidence = _build_evidence()

    assert "measurements" in evidence.metadata


def test_measurements_are_named():
    _, evidence = _build_evidence()

    measurements = evidence.metadata["measurements"]

    assert set(measurements) == {
        "generalization_gap",
        "training_accuracy",
        "evaluation_accuracy",
    }


def test_measurements_use_evidence_measurement_type():
    _, evidence = _build_evidence()

    measurements = evidence.metadata["measurements"]

    assert all(
        isinstance(
            measurement,
            EvidenceMeasurement,
        )
        for measurement in measurements.values()
    )


def test_generalization_gap_measurement_matches_value():
    _, evidence = _build_evidence()

    measurement = evidence.metadata["measurements"][
        "generalization_gap"
    ]

    assert measurement.value == pytest.approx(
        evidence.value
    )


def test_training_accuracy_measurement_is_correct():
    result, evidence = _build_evidence()

    measurement = evidence.metadata["measurements"][
        "training_accuracy"
    ]

    assert measurement.value == pytest.approx(
        result.train_accuracy
    )


def test_evaluation_accuracy_measurement_is_correct():
    result, evidence = _build_evidence()

    measurement = evidence.metadata["measurements"][
        "evaluation_accuracy"
    ]

    assert measurement.value == pytest.approx(
        result.test_accuracy
    )


def test_g2_still_uses_generalization_gap_as_primary_value():
    result, evidence = _build_evidence()

    expected_gap = (
        result.train_accuracy
        - result.test_accuracy
    )

    assert evidence.value == pytest.approx(
        expected_gap
    )


def test_measurement_names_match_their_values():
    _, evidence = _build_evidence()

    measurements = evidence.metadata["measurements"]

    assert (
        measurements["generalization_gap"].name
        == "generalization_gap"
    )

    assert (
        measurements["training_accuracy"].name
        == "training_accuracy"
    )

    assert (
        measurements["evaluation_accuracy"].name
        == "evaluation_accuracy"
    )