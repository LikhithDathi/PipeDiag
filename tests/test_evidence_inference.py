import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_inference import EvidenceInference


def create_inference():
    return BayesianInference(
        hypothesis_ids=["H1", "H2", "H3"],
        priors={
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        },
    )


def test_no_evidence_returns_prior_distribution():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    posterior = inference.infer([])

    assert posterior == pytest.approx(
        {
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        }
    )


def test_single_evidence_group_updates_posterior():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = EvidenceGroup(
        name="generalization",
        value=0.8,
        status="observed",
    )

    likelihoods = {
        "generalization": {
            "H1": 0.1,
            "H2": 0.8,
            "H3": 0.1,
        }
    }

    posterior = inference.infer(
        [evidence],
        likelihoods,
    )

    assert posterior["H2"] > posterior["H1"]
    assert posterior["H2"] > posterior["H3"]


def test_multiple_evidence_groups_are_combined():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="evidence_a",
            value=0.8,
            status="observed",
        ),
        EvidenceGroup(
            name="evidence_b",
            value=0.9,
            status="observed",
        ),
    ]

    likelihoods = {
        "evidence_a": {
            "H1": 0.8,
            "H2": 0.1,
            "H3": 0.1,
        },
        "evidence_b": {
            "H1": 0.9,
            "H2": 0.05,
            "H3": 0.05,
        },
    }

    posterior = inference.infer(
        evidence,
        likelihoods,
    )

    assert posterior["H1"] > posterior["H2"]
    assert posterior["H1"] > posterior["H3"]


def test_unavailable_evidence_is_ignored():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="available",
            value=0.8,
            status="observed",
        ),
        EvidenceGroup(
            name="missing",
            value=None,
            status="unavailable",
        ),
    ]

    likelihoods = {
        "available": {
            "H1": 0.8,
            "H2": 0.1,
            "H3": 0.1,
        },
        "missing": {
            "H1": 0.01,
            "H2": 0.98,
            "H3": 0.01,
        },
    }

    posterior = inference.infer(
        evidence,
        likelihoods,
    )

    assert posterior["H1"] > posterior["H2"]
    assert posterior["H1"] > posterior["H3"]


def test_unavailable_evidence_does_not_change_prior_when_alone():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="missing",
            value=None,
            status="unavailable",
        ),
    ]

    likelihoods = {
        "missing": {
            "H1": 0.01,
            "H2": 0.98,
            "H3": 0.01,
        }
    }

    posterior = inference.infer(
        evidence,
        likelihoods,
    )

    assert posterior == pytest.approx(
        {
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        }
    )


def test_unknown_evidence_status_is_rejected():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="generalization",
            value=0.8,
            status="unknown",
        ),
    ]

    likelihoods = {
        "generalization": {
            "H1": 0.1,
            "H2": 0.8,
            "H3": 0.1,
        }
    }

    with pytest.raises(ValueError):
        inference.infer(
            evidence,
            likelihoods,
        )


def test_missing_likelihood_model_for_observed_evidence_is_rejected():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="generalization",
            value=0.8,
            status="observed",
        ),
    ]

    with pytest.raises(ValueError):
        inference.infer(
            evidence,
            likelihoods={},
        )


def test_unknown_evidence_likelihood_model_is_rejected():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="generalization",
            value=0.8,
            status="observed",
        ),
    ]

    likelihoods = {
        "other": {
            "H1": 0.5,
            "H2": 0.3,
            "H3": 0.2,
        }
    }

    with pytest.raises(ValueError):
        inference.infer(
            evidence,
            likelihoods,
        )


def test_observed_evidence_requires_complete_hypothesis_likelihoods():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="generalization",
            value=0.8,
            status="observed",
        ),
    ]

    likelihoods = {
        "generalization": {
            "H1": 0.1,
            "H2": 0.8,
        }
    }

    with pytest.raises(ValueError):
        inference.infer(
            evidence,
            likelihoods,
        )


def test_posterior_probabilities_sum_to_one():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="generalization",
            value=0.8,
            status="observed",
        ),
    ]

    likelihoods = {
        "generalization": {
            "H1": 0.2,
            "H2": 0.7,
            "H3": 0.1,
        }
    }

    posterior = inference.infer(
        evidence,
        likelihoods,
    )

    assert sum(posterior.values()) == pytest.approx(1.0)


def test_duplicate_evidence_group_names_are_rejected():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="generalization",
            value=0.8,
            status="observed",
        ),
        EvidenceGroup(
            name="generalization",
            value=0.9,
            status="observed",
        ),
    ]

    likelihoods = {
        "generalization": {
            "H1": 0.2,
            "H2": 0.7,
            "H3": 0.1,
        }
    }

    with pytest.raises(ValueError):
        inference.infer(
            evidence,
            likelihoods,
        )


def test_likelihoods_for_unavailable_evidence_are_not_required():
    inference = EvidenceInference(
        bayesian_inference=create_inference()
    )

    evidence = [
        EvidenceGroup(
            name="available",
            value=0.8,
            status="observed",
        ),
        EvidenceGroup(
            name="missing",
            value=None,
            status="unavailable",
        ),
    ]

    likelihoods = {
        "available": {
            "H1": 0.7,
            "H2": 0.2,
            "H3": 0.1,
        }
    }

    posterior = inference.infer(
        evidence,
        likelihoods,
    )

    assert posterior["H1"] > posterior["H2"]