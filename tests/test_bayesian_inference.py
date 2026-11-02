import math

import pytest

from pipediag.diagnostics.bayesian import BayesianInference


def test_uniform_priors_are_created_for_nine_hypotheses():
    inference = BayesianInference(
        hypothesis_ids=[
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
            "H7",
            "H8",
            "H9",
        ]
    )

    priors = inference.priors

    assert len(priors) == 9

    for probability in priors.values():
        assert probability == pytest.approx(1 / 9)


def test_custom_priors_are_preserved():
    priors = {
        "H1": 0.5,
        "H2": 0.3,
        "H3": 0.2,
    }

    inference = BayesianInference(
        hypothesis_ids=["H1", "H2", "H3"],
        priors=priors,
    )

    assert inference.priors == priors


def test_priors_must_sum_to_one():
    with pytest.raises(ValueError):
        BayesianInference(
            hypothesis_ids=["H1", "H2"],
            priors={
                "H1": 0.5,
                "H2": 0.4,
            },
        )


def test_priors_must_be_non_negative():
    with pytest.raises(ValueError):
        BayesianInference(
            hypothesis_ids=["H1", "H2"],
            priors={
                "H1": -0.1,
                "H2": 1.1,
            },
        )


def test_posterior_probabilities_sum_to_one():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
        priors={
            "H1": 0.5,
            "H2": 0.5,
        },
    )

    posterior = inference.infer(
        likelihoods={
            "H1": 0.8,
            "H2": 0.2,
        }
    )

    assert sum(posterior.values()) == pytest.approx(1.0)


def test_higher_likelihood_produces_higher_posterior_support():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
        priors={
            "H1": 0.5,
            "H2": 0.5,
        },
    )

    posterior = inference.infer(
        likelihoods={
            "H1": 0.9,
            "H2": 0.1,
        }
    )

    assert posterior["H1"] > posterior["H2"]


def test_prior_probability_affects_posterior():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
        priors={
            "H1": 0.9,
            "H2": 0.1,
        },
    )

    posterior = inference.infer(
        likelihoods={
            "H1": 0.5,
            "H2": 0.5,
        }
    )

    assert posterior["H1"] == pytest.approx(0.9)
    assert posterior["H2"] == pytest.approx(0.1)


def test_incomplete_likelihood_vector_is_rejected():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
        priors={
            "H1": 0.5,
            "H2": 0.5,
        },
    )

    with pytest.raises(ValueError):
        inference.infer(
            likelihoods={
                "H1": 0.8,
            }
        )


def test_empty_likelihood_vector_is_rejected():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
    )

    with pytest.raises(ValueError):
        inference.infer(likelihoods={})


def test_posterior_values_are_finite_and_bounded():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
        priors={
            "H1": 0.5,
            "H2": 0.5,
        },
    )

    posterior = inference.infer(
        likelihoods={
            "H1": 0.8,
            "H2": 0.2,
        }
    )

    for probability in posterior.values():
        assert math.isfinite(probability)
        assert 0.0 <= probability <= 1.0


def test_unknown_hypothesis_in_likelihoods_is_rejected():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
    )

    with pytest.raises(ValueError):
        inference.infer(
            likelihoods={
                "H1": 0.5,
                "H3": 0.5,
            }
        )


def test_negative_likelihood_is_rejected():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
    )

    with pytest.raises(ValueError):
        inference.infer(
            likelihoods={
                "H1": -0.1,
                "H2": 0.5,
            }
        )


def test_likelihood_above_one_is_rejected():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
    )

    with pytest.raises(ValueError):
        inference.infer(
            likelihoods={
                "H1": 1.2,
                "H2": 0.5,
            }
        )


def test_all_zero_likelihoods_are_rejected():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
    )

    with pytest.raises(ValueError):
        inference.infer(
            likelihoods={
                "H1": 0.0,
                "H2": 0.0,
            }
        )


def test_inference_does_not_modify_input_likelihoods():
    inference = BayesianInference(
        hypothesis_ids=["H1", "H2"],
    )

    likelihoods = {
        "H1": 0.8,
        "H2": 0.2,
    }

    original = likelihoods.copy()

    inference.infer(likelihoods)

    assert likelihoods == original