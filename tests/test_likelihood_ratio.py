import pytest

from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)


def _model():
    return GaussianLikelihoodModel(
        mean_true=0.90,
        std_true=0.05,
        mean_false=0.10,
        std_false=0.05,
    )


def test_likelihood_ratio_is_positive():
    model = _model()

    ratio = model.likelihood_ratio(0.90)

    assert ratio > 0


def test_evidence_near_true_mean_supports_hypothesis():
    model = _model()

    ratio = model.likelihood_ratio(0.90)

    assert ratio > 1.0


def test_evidence_near_false_mean_supports_alternative():
    model = _model()

    ratio = model.likelihood_ratio(0.10)

    assert ratio < 1.0


def test_equal_distributions_produce_ratio_one():
    model = GaussianLikelihoodModel(
        mean_true=0.50,
        std_true=0.10,
        mean_false=0.50,
        std_false=0.10,
    )

    ratio = model.likelihood_ratio(0.50)

    assert ratio == pytest.approx(1.0)


def test_likelihood_ratio_matches_density_ratio():
    model = _model()

    evidence_value = 0.75

    true_likelihood = model.likelihood(
        evidence_value=evidence_value,
        hypothesis_true=True,
    )

    false_likelihood = model.likelihood(
        evidence_value=evidence_value,
        hypothesis_true=False,
    )

    ratio = model.likelihood_ratio(evidence_value)

    assert ratio == pytest.approx(
        true_likelihood / false_likelihood
    )


def test_non_finite_evidence_is_rejected():
    model = _model()

    with pytest.raises(ValueError):
        model.likelihood_ratio(float("nan"))