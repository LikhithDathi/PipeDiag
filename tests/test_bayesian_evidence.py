import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)


HYPOTHESES = ["H1", "H2", "H3"]


def create_inference():
    return BayesianEvidenceInference(
        BayesianInference(HYPOTHESES)
    )


def test_no_evidence_returns_prior():
    inference = create_inference()

    result = inference.infer({})

    assert result == pytest.approx(
        {
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        }
    )


def test_single_evidence_group_updates_posterior():
    inference = create_inference()

    result = inference.infer(
        {
            "G1": {
                "H1": 4.0,
            }
        }
    )

    assert result["H1"] > result["H2"]
    assert result["H1"] > result["H3"]


def test_posterior_sums_to_one():
    inference = create_inference()

    result = inference.infer(
        {
            "G1": {
                "H1": 0.8,
            },
            "G2": {
                "H1": 0.2,
                "H2": 0.7,
                "H3": 0.1,
            },
        }
    )

    assert sum(result.values()) == pytest.approx(1.0)


def test_multiple_evidence_groups_are_combined():
    inference = create_inference()

    result = inference.infer(
        {
            "G1": {
                "H1": 0.8,
            },
            "G2": {
                "H2": 0.7,
            },
        }
    )

    assert result["H1"] > 0.0
    assert result["H2"] > 0.0
    assert result["H3"] > 0.0


def test_likelihood_densities_can_be_greater_than_one():
    inference = create_inference()

    result = inference.infer(
        {
            "G1": {
                "H1": 4.0,
            }
        }
    )

    assert result["H1"] > result["H2"]
    assert sum(result.values()) == pytest.approx(1.0)


def test_unavailable_evidence_is_ignored():
    inference = create_inference()

    result_without = inference.infer(
        {
            "G1": {
                "H1": 0.8,
            }
        }
    )

    result_with_unavailable = inference.infer(
        {
            "G1": {
                "H1": 0.8,
            },
            "G2": None,
        }
    )

    assert result_with_unavailable == pytest.approx(
        result_without
    )


def test_missing_hypothesis_likelihood_is_neutral():
    inference = create_inference()

    result = inference.infer(
        {
            "G1": {
                "H1": 0.8,
            }
        }
    )

    assert result["H2"] > 0.0
    assert result["H3"] > 0.0


def test_unknown_hypothesis_is_rejected():
    inference = create_inference()

    with pytest.raises(ValueError):
        inference.infer(
            {
                "G1": {
                    "H99": 0.1,
                }
            }
        )


def test_empty_evidence_group_is_rejected():
    inference = create_inference()

    with pytest.raises(ValueError):
        inference.infer(
            {
                "G1": {}
            }
        )


def test_negative_likelihood_is_rejected():
    inference = create_inference()

    with pytest.raises(ValueError):
        inference.infer(
            {
                "G1": {
                    "H1": -0.1,
                }
            }
        )


def test_non_finite_likelihood_is_rejected():
    inference = create_inference()

    with pytest.raises(ValueError):
        inference.infer(
            {
                "G1": {
                    "H1": float("nan"),
                }
            }
        )


def test_zero_total_probability_is_rejected():
    inference = create_inference()

    with pytest.raises(ValueError):
        inference.infer(
            {
                "G1": {
                    "H1": 0.0,
                    "H2": 0.0,
                    "H3": 0.0,
                }
            }
        )


def test_combined_evidence_can_shift_the_leading_hypothesis():
    inference = create_inference()

    result = inference.infer(
        {
            "G1": {
                "H1": 0.9,
            },
            "G2": {
                "H2": 0.9,
            },
        }
    )

    assert result["H1"] == pytest.approx(
        result["H2"]
    )


def test_prior_is_used_when_all_evidence_is_unavailable():
    inference = create_inference()

    result = inference.infer(
        {
            "G1": None,
            "G2": None,
        }
    )

    assert result == pytest.approx(
        {
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        }
    )