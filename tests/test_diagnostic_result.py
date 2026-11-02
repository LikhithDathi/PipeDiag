import pytest

from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.evidence_group import EvidenceGroup


def create_posterior():
    return {
        "H1": 0.1,
        "H2": 0.7,
        "H3": 0.05,
        "H4": 0.03,
        "H5": 0.02,
        "H6": 0.02,
        "H7": 0.04,
        "H8": 0.02,
        "H9": 0.02,
    }


def test_diagnostic_result_stores_posterior():
    posterior = create_posterior()

    result = DiagnosticResult(
        posterior=posterior,
        uncertainty=0.42,
        evidence=[],
    )

    assert result.posterior == posterior


def test_diagnostic_result_stores_uncertainty():
    result = DiagnosticResult(
        posterior=create_posterior(),
        uncertainty=0.42,
        evidence=[],
    )

    assert result.uncertainty == 0.42


def test_diagnostic_result_stores_evidence():
    evidence = [
        EvidenceGroup(
            name="generalization",
            value=0.31,
            status="observed",
        ),
    ]

    result = DiagnosticResult(
        posterior=create_posterior(),
        uncertainty=0.42,
        evidence=evidence,
    )

    assert result.evidence == evidence


def test_diagnostic_result_defaults_metadata_to_empty_dictionary():
    result = DiagnosticResult(
        posterior=create_posterior(),
        uncertainty=0.42,
        evidence=[],
    )

    assert result.metadata == {}


def test_diagnostic_result_can_store_metadata():
    result = DiagnosticResult(
        posterior=create_posterior(),
        uncertainty=0.42,
        evidence=[],
        metadata={
            "run_id": "run-123",
            "pipeline": "classification",
        },
    )

    assert result.metadata["run_id"] == "run-123"
    assert result.metadata["pipeline"] == "classification"


def test_posterior_must_sum_to_one():
    posterior = create_posterior()
    posterior["H2"] = 0.5

    with pytest.raises(ValueError):
        DiagnosticResult(
            posterior=posterior,
            uncertainty=0.42,
            evidence=[],
        )


def test_posterior_probabilities_must_be_bounded():
    posterior = create_posterior()
    posterior["H2"] = 1.2

    with pytest.raises(ValueError):
        DiagnosticResult(
            posterior=posterior,
            uncertainty=0.42,
            evidence=[],
        )


def test_uncertainty_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        DiagnosticResult(
            posterior=create_posterior(),
            uncertainty=1.2,
            evidence=[],
        )


def test_negative_uncertainty_is_rejected():
    with pytest.raises(ValueError):
        DiagnosticResult(
            posterior=create_posterior(),
            uncertainty=-0.1,
            evidence=[],
        )


def test_empty_posterior_is_rejected():
    with pytest.raises(ValueError):
        DiagnosticResult(
            posterior={},
            uncertainty=0.0,
            evidence=[],
        )


def test_diagnostic_result_is_immutable():
    result = DiagnosticResult(
        posterior=create_posterior(),
        uncertainty=0.42,
        evidence=[],
    )

    with pytest.raises(AttributeError):
        result.uncertainty = 0.1