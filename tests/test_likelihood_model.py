import math

import pytest

from pipediag.diagnostics.likelihood_model import GaussianLikelihoodModel


def test_gaussian_likelihood_model_stores_parameters():
    model = GaussianLikelihoodModel(
        mean_true=0.8,
        std_true=0.1,
        mean_false=0.2,
        std_false=0.1,
    )

    assert model.mean_true == 0.8
    assert model.std_true == 0.1
    assert model.mean_false == 0.2
    assert model.std_false == 0.1


def test_gaussian_likelihood_for_true_hypothesis_is_positive():
    model = GaussianLikelihoodModel(
        mean_true=0.8,
        std_true=0.1,
        mean_false=0.2,
        std_false=0.1,
    )

    likelihood = model.likelihood(0.8, hypothesis_true=True)

    assert likelihood > 0.0
    assert math.isfinite(likelihood)


def test_gaussian_likelihood_for_false_hypothesis_is_positive():
    model = GaussianLikelihoodModel(
        mean_true=0.8,
        std_true=0.1,
        mean_false=0.2,
        std_false=0.1,
    )

    likelihood = model.likelihood(0.2, hypothesis_true=False)

    assert likelihood > 0.0
    assert math.isfinite(likelihood)


def test_evidence_near_true_mean_has_higher_true_likelihood():
    model = GaussianLikelihoodModel(
        mean_true=0.8,
        std_true=0.1,
        mean_false=0.2,
        std_false=0.1,
    )

    near_true = model.likelihood(0.8, hypothesis_true=True)
    far_from_true = model.likelihood(0.2, hypothesis_true=True)

    assert near_true > far_from_true


def test_evidence_near_false_mean_has_higher_false_likelihood():
    model = GaussianLikelihoodModel(
        mean_true=0.8,
        std_true=0.1,
        mean_false=0.2,
        std_false=0.1,
    )

    near_false = model.likelihood(0.2, hypothesis_true=False)
    far_from_false = model.likelihood(0.8, hypothesis_true=False)

    assert near_false > far_from_false


def test_standard_deviation_must_be_positive():
    with pytest.raises(ValueError):
        GaussianLikelihoodModel(
            mean_true=0.8,
            std_true=0.0,
            mean_false=0.2,
            std_false=0.1,
        )


def test_false_standard_deviation_must_be_positive():
    with pytest.raises(ValueError):
        GaussianLikelihoodModel(
            mean_true=0.8,
            std_true=0.1,
            mean_false=0.2,
            std_false=0.0,
        )


def test_likelihood_requires_finite_evidence():
    model = GaussianLikelihoodModel(
        mean_true=0.8,
        std_true=0.1,
        mean_false=0.2,
        std_false=0.1,
    )

    with pytest.raises(ValueError):
        model.likelihood(float("nan"), hypothesis_true=True)


def test_likelihood_model_is_immutable():
    model = GaussianLikelihoodModel(
        mean_true=0.8,
        std_true=0.1,
        mean_false=0.2,
        std_false=0.1,
    )

    with pytest.raises(AttributeError):
        model.mean_true = 0.5