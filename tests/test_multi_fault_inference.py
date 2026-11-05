
import math

import pytest

from pipediag.diagnostics.multi_fault_bayesian import (
    MultiFaultBayesianInference,
)


@pytest.fixture
def inference():
    return MultiFaultBayesianInference(
        hypothesis_ids=("H1", "H2", "H3"),
        priors={
            "H1": 0.10,
            "H2": 0.10,
            "H3": 0.10,
        },
    )


def test_normalized_probabilities_sum_to_one(inference):
    posterior = inference.infer_with_log_likelihood_ratios(
        {
            "G3": {"H1": 4.0},
            "G4": {"H2": 3.0},
        }
    )

    assert sum(posterior.values()) == pytest.approx(1.0)


def test_only_hypotheses_with_eligible_evidence_are_normalized(
    inference,
):
    posterior = inference.infer_with_log_likelihood_ratios(
        {
            "G3": {"H1": 4.0},
            "G4": {"H2": 3.0},
        }
    )

    assert set(posterior) == {"H1", "H2"}
    assert "H3" not in posterior


def test_multiple_hypotheses_receive_normalized_support(
    inference,
):
    posterior = inference.infer_with_log_likelihood_ratios(
        {
            "G3": {"H1": 4.0},
            "G4": {"H2": 3.0},
        }
    )

    assert posterior["H1"] > 0.5
    assert posterior["H2"] < 0.5
    assert posterior["H1"] + posterior["H2"] == pytest.approx(1.0)


def test_independent_probabilities_are_preserved(inference):
    independent = inference.infer_independent_probabilities(
        {
            "G3": {"H1": 4.0},
            "G4": {"H2": 3.0},
        }
    )

    assert independent["H1"] > 0.5
    assert independent["H2"] > 0.5
    assert independent["H1"] + independent["H2"] != pytest.approx(1.0)


def test_positive_evidence_increases_independent_probability(
    inference,
):
    independent = inference.infer_independent_probabilities(
        {
            "G3": {"H1": 2.0},
        }
    )

    assert independent["H1"] > 0.10


def test_negative_evidence_decreases_independent_probability(
    inference,
):
    independent = inference.infer_independent_probabilities(
        {
            "G3": {"H1": -2.0},
        }
    )

    assert independent["H1"] < 0.10


def test_normalized_support_respects_relative_independent_support(
    inference,
):
    independent = inference.infer_independent_probabilities(
        {
            "G3": {"H1": 2.0},
            "G4": {"H2": -2.0},
        }
    )

    normalized = inference.infer_with_log_likelihood_ratios(
        {
            "G3": {"H1": 2.0},
            "G4": {"H2": -2.0},
        }
    )

    assert normalized["H1"] > normalized["H2"]
    assert normalized["H1"] == pytest.approx(
        independent["H1"]
        / (independent["H1"] + independent["H2"])
    )


def test_empty_evidence_returns_normalized_priors(inference):
    posterior = inference.infer_with_log_likelihood_ratios({})

    assert set(posterior) == {"H1", "H2", "H3"}
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert all(
        probability == pytest.approx(1.0 / 3.0)
        for probability in posterior.values()
    )


def test_extreme_log_ratios_remain_numerically_stable(
    inference,
):
    posterior = inference.infer_with_log_likelihood_ratios(
        {
            "G3": {"H1": 1000.0},
            "G4": {"H2": -1000.0},
        }
    )

    assert posterior["H1"] == pytest.approx(1.0)
    assert posterior["H2"] == pytest.approx(0.0)

    for probability in posterior.values():
        assert math.isfinite(probability)
        assert 0.0 <= probability <= 1.0

    assert sum(posterior.values()) == pytest.approx(1.0)


def test_unknown_hypothesis_is_rejected(inference):
    with pytest.raises(ValueError):
        inference.infer_with_log_likelihood_ratios(
            {
                "G3": {"H99": 2.0},
            }
        )


def test_non_finite_log_ratio_is_rejected(inference):
    with pytest.raises(ValueError):
        inference.infer_with_log_likelihood_ratios(
            {
                "G3": {"H1": float("nan")},
            }
        )


def test_invalid_prior_is_rejected():
    with pytest.raises(ValueError):
        MultiFaultBayesianInference(
            hypothesis_ids=("H1", "H2"),
            priors={
                "H1": 1.0,
                "H2": 0.0,
            },
        )
