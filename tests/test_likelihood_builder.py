import pytest

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.likelihood_builder import (
    LikelihoodModelBuilder,
)
from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)


def _observations():
    return [
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.80,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.85,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.90,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.10,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.15,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.20,
        ),
    ]


def test_builder_creates_registry():
    builder = LikelihoodModelBuilder()

    registry = builder.build(_observations())

    assert registry.contains("G1", "H1")


def test_builder_creates_gaussian_model():
    builder = LikelihoodModelBuilder()

    registry = builder.build(_observations())

    model = registry.get("G1", "H1")

    assert isinstance(model, GaussianLikelihoodModel)


def test_builder_uses_observed_true_mean():
    builder = LikelihoodModelBuilder()

    registry = builder.build(_observations())

    model = registry.get("G1", "H1")

    assert model.mean_true == pytest.approx(0.85)


def test_builder_uses_observed_false_mean():
    builder = LikelihoodModelBuilder()

    registry = builder.build(_observations())

    model = registry.get("G1", "H1")

    assert model.mean_false == pytest.approx(0.15)


def test_builder_produces_positive_standard_deviations():
    builder = LikelihoodModelBuilder()

    registry = builder.build(_observations())

    model = registry.get("G1", "H1")

    assert model.std_true > 0
    assert model.std_false > 0


def test_builder_supports_multiple_evidence_hypothesis_pairs():
    observations = _observations() + [
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=True,
            value=0.70,
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=True,
            value=0.75,
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=False,
            value=0.20,
        ),
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H2",
            hypothesis_true=False,
            value=0.25,
        ),
    ]

    builder = LikelihoodModelBuilder()

    registry = builder.build(observations)

    assert registry.contains("G1", "H1")
    assert registry.contains("G2", "H2")


def test_builder_rejects_empty_observations():
    builder = LikelihoodModelBuilder()

    with pytest.raises(ValueError):
        builder.build([])


def test_builder_rejects_missing_true_observations():
    observations = [
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.10,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.20,
        ),
    ]

    builder = LikelihoodModelBuilder()

    with pytest.raises(ValueError):
        builder.build(observations)


def test_builder_rejects_missing_false_observations():
    observations = [
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.80,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.90,
        ),
    ]

    builder = LikelihoodModelBuilder()

    with pytest.raises(ValueError):
        builder.build(observations)