import pytest

from pipediag.diagnostics.likelihood_builder import (
    LikelihoodModelBuilder,
)

from experiments.generalization_calibration import (
    collect_generalization_calibration_observations,
)


def _build_model():
    observations = (
        collect_generalization_calibration_observations(
            n_samples=500,
            n_features=10,
            random_states=(1, 2, 3, 4, 5),
        )
    )

    registry = LikelihoodModelBuilder().build(
        observations
    )

    return registry.get("G2", "H2")


def test_g2_h2_model_is_registered():
    model = _build_model()

    assert model is not None


def test_true_and_false_means_are_finite():
    model = _build_model()

    assert model.mean_true == pytest.approx(
        float(model.mean_true)
    )

    assert model.mean_false == pytest.approx(
        float(model.mean_false)
    )


def test_true_distribution_has_positive_variance():
    model = _build_model()

    assert model.std_true > 0


def test_false_distribution_has_positive_variance():
    model = _build_model()

    assert model.std_false > 0


def test_generalization_failure_has_larger_mean_gap():
    model = _build_model()

    assert model.mean_true > model.mean_false


def test_large_gap_has_likelihood_ratio_above_one():
    model = _build_model()

    evidence_value = model.mean_true

    ratio = model.likelihood_ratio(
        evidence_value
    )

    assert ratio > 1.0


def test_small_gap_has_likelihood_ratio_below_one():
    model = _build_model()

    evidence_value = model.mean_false

    ratio = model.likelihood_ratio(
        evidence_value
    )

    assert ratio < 1.0


def test_likelihood_ratio_is_finite():
    model = _build_model()

    ratio = model.likelihood_ratio(
        model.mean_true
    )

    assert ratio == pytest.approx(
        float(ratio)
    )


def test_model_discriminates_h2_evidence():
    model = _build_model()

    true_ratio = model.likelihood_ratio(
        model.mean_true
    )

    false_ratio = model.likelihood_ratio(
        model.mean_false
    )

    assert true_ratio > false_ratio