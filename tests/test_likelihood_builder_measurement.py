from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.likelihood_builder import (
    LikelihoodModelBuilder,
)


def test_builder_preserves_measurement():
    observations = [
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=True,
            value=0.30,
            measurement="generalization_gap",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=True,
            value=0.35,
            measurement="generalization_gap",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=False,
            value=0.05,
            measurement="generalization_gap",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=False,
            value=0.10,
            measurement="generalization_gap",
        ),
    ]

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G2", "H2")

    assert model.measurement == (
        "generalization_gap"
    )


def test_builder_preserves_training_accuracy_measurement():
    observations = [
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=True,
            value=0.70,
            measurement="training_accuracy",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=True,
            value=0.75,
            measurement="training_accuracy",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=False,
            value=0.90,
            measurement="training_accuracy",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=False,
            value=0.95,
            measurement="training_accuracy",
        ),
    ]

    registry = LikelihoodModelBuilder().build(
        observations
    )

    model = registry.get("G2", "H7")

    assert model.measurement == (
        "training_accuracy"
    )


def test_h2_and_h7_can_use_different_measurements():
    observations = [
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=True,
            value=0.30,
            measurement="generalization_gap",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=True,
            value=0.35,
            measurement="generalization_gap",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=False,
            value=0.05,
            measurement="generalization_gap",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=False,
            value=0.10,
            measurement="generalization_gap",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=True,
            value=0.70,
            measurement="training_accuracy",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=True,
            value=0.75,
            measurement="training_accuracy",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=False,
            value=0.90,
            measurement="training_accuracy",
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H7",
            hypothesis_true=False,
            value=0.95,
            measurement="training_accuracy",
        ),
    ]

    registry = LikelihoodModelBuilder().build(
        observations
    )

    assert registry.get(
        "G2", "H2"
    ).measurement == "generalization_gap"

    assert registry.get(
        "G2", "H7"
    ).measurement == "training_accuracy"