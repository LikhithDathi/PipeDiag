import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)


def _inference():
    bayesian = BayesianInference(
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

    return BayesianEvidenceInference(bayesian)


def test_ratio_greater_than_one_increases_hypothesis_support():
    inference = _inference()

    posterior = inference.infer_with_likelihood_ratios(
        {
            "G1": {
                "H1": 10.0,
            }
        }
    )

    assert posterior["H1"] > 1.0 / 9.0


def test_ratio_less_than_one_decreases_hypothesis_support():
    inference = _inference()

    posterior = inference.infer_with_likelihood_ratios(
        {
            "G1": {
                "H1": 0.1,
            }
        }
    )

    assert posterior["H1"] < 1.0 / 9.0


def test_ratio_equal_to_one_is_neutral():
    inference = _inference()

    posterior = inference.infer_with_likelihood_ratios(
        {
            "G1": {
                "H1": 1.0,
            }
        }
    )

    assert posterior["H1"] == pytest.approx(
        1.0 / 9.0
    )


def test_unmapped_hypotheses_remain_neutral():
    inference = _inference()

    posterior = inference.infer_with_likelihood_ratios(
        {
            "G1": {
                "H1": 10.0,
            }
        }
    )

    assert posterior["H2"] == pytest.approx(
        posterior["H3"]
    )


def test_posterior_is_normalized():
    inference = _inference()

    posterior = inference.infer_with_likelihood_ratios(
        {
            "G1": {
                "H1": 10.0,
            }
        }
    )

    assert sum(posterior.values()) == pytest.approx(1.0)


def test_multiple_evidence_groups_combine():
    inference = _inference()

    posterior = inference.infer_with_likelihood_ratios(
        {
            "G1": {
                "H1": 5.0,
            },
            "G2": {
                "H2": 3.0,
            },
        }
    )

    assert posterior["H1"] > 1.0 / 9.0
    assert posterior["H2"] > 1.0 / 9.0


def test_empty_ratios_preserve_prior():
    inference = _inference()

    posterior = inference.infer_with_likelihood_ratios({})

    assert posterior["H1"] == pytest.approx(
        1.0 / 9.0
    )


def test_none_ratios_preserve_prior():
    inference = _inference()

    posterior = inference.infer_with_likelihood_ratios(
        {
            "G1": None,
        }
    )

    assert posterior["H1"] == pytest.approx(
        1.0 / 9.0
    )