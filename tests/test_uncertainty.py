import math

import pytest

from pipediag.diagnostics.uncertainty import posterior_uncertainty


def test_uniform_distribution_has_maximum_uncertainty():
    posterior = {
        "H1": 1 / 9,
        "H2": 1 / 9,
        "H3": 1 / 9,
        "H4": 1 / 9,
        "H5": 1 / 9,
        "H6": 1 / 9,
        "H7": 1 / 9,
        "H8": 1 / 9,
        "H9": 1 / 9,
    }

    uncertainty = posterior_uncertainty(posterior)

    assert uncertainty == pytest.approx(1.0)


def test_single_dominant_hypothesis_has_zero_uncertainty():
    posterior = {
        "H1": 1.0,
        "H2": 0.0,
        "H3": 0.0,
        "H4": 0.0,
        "H5": 0.0,
        "H6": 0.0,
        "H7": 0.0,
        "H8": 0.0,
        "H9": 0.0,
    }

    uncertainty = posterior_uncertainty(posterior)

    assert uncertainty == pytest.approx(0.0)


def test_uncertainty_is_between_zero_and_one():
    posterior = {
        "H1": 0.6,
        "H2": 0.2,
        "H3": 0.1,
        "H4": 0.1,
    }

    uncertainty = posterior_uncertainty(posterior)

    assert 0.0 <= uncertainty <= 1.0


def test_more_concentrated_distribution_has_lower_uncertainty():
    concentrated = {
        "H1": 0.8,
        "H2": 0.1,
        "H3": 0.1,
    }

    distributed = {
        "H1": 0.34,
        "H2": 0.33,
        "H3": 0.33,
    }

    concentrated_uncertainty = posterior_uncertainty(
        concentrated
    )

    distributed_uncertainty = posterior_uncertainty(
        distributed
    )

    assert concentrated_uncertainty < distributed_uncertainty


def test_uncertainty_is_symmetric_to_hypothesis_order():
    posterior_a = {
        "H1": 0.7,
        "H2": 0.2,
        "H3": 0.1,
    }

    posterior_b = {
        "H1": 0.1,
        "H2": 0.7,
        "H3": 0.2,
    }

    assert posterior_uncertainty(
        posterior_a
    ) == pytest.approx(
        posterior_uncertainty(posterior_b)
    )


def test_probabilities_must_sum_to_one():
    posterior = {
        "H1": 0.6,
        "H2": 0.2,
        "H3": 0.1,
    }

    with pytest.raises(ValueError):
        posterior_uncertainty(posterior)


def test_negative_probability_is_rejected():
    posterior = {
        "H1": 0.8,
        "H2": 0.3,
        "H3": -0.1,
    }

    with pytest.raises(ValueError):
        posterior_uncertainty(posterior)


def test_probability_above_one_is_rejected():
    posterior = {
        "H1": 1.2,
        "H2": -0.1,
    }

    with pytest.raises(ValueError):
        posterior_uncertainty(posterior)


def test_empty_posterior_is_rejected():
    with pytest.raises(ValueError):
        posterior_uncertainty({})


def test_non_finite_probability_is_rejected():
    posterior = {
        "H1": float("nan"),
        "H2": 0.5,
        "H3": 0.5,
    }

    with pytest.raises(ValueError):
        posterior_uncertainty(posterior)


def test_uncertainty_is_finite():
    posterior = {
        "H1": 0.5,
        "H2": 0.3,
        "H3": 0.2,
    }

    uncertainty = posterior_uncertainty(posterior)

    assert math.isfinite(uncertainty)