import pytest

from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)


def _build_model():
    return GaussianLikelihoodModel(
        mean_true=0.2,
        std_true=0.05,
        mean_false=0.05,
        std_false=0.02,
        measurement="generalization_gap",
    )


def test_measurement_is_stored():
    model = _build_model()

    assert model.measurement == (
        "generalization_gap"
    )


def test_training_accuracy_measurement_is_supported():
    model = GaussianLikelihoodModel(
        mean_true=0.70,
        std_true=0.05,
        mean_false=0.90,
        std_false=0.03,
        measurement="training_accuracy",
    )

    assert model.measurement == (
        "training_accuracy"
    )


def test_empty_measurement_is_rejected():
    with pytest.raises(ValueError):
        GaussianLikelihoodModel(
            mean_true=0.2,
            std_true=0.05,
            mean_false=0.05,
            std_false=0.02,
            measurement="",
        )


def test_default_measurement_is_supported():
    model = GaussianLikelihoodModel(
        mean_true=0.2,
        std_true=0.05,
        mean_false=0.05,
        std_false=0.02,
    )

    assert model.measurement == "default"


def test_likelihood_behavior_is_unchanged():
    model = _build_model()

    likelihood = model.likelihood(
        evidence_value=0.2,
        hypothesis_true=True,
    )

    assert likelihood > 0


def test_likelihood_ratio_behavior_is_unchanged():
    model = _build_model()

    ratio = model.likelihood_ratio(0.2)

    assert ratio > 1.0