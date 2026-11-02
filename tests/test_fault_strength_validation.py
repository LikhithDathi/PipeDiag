import math

import pytest

from experiments.fault_strength_validation import (
    EVIDENCE_DIRECTIONS,
    FaultStrengthObservation,
    FaultStrengthValidation,
    run_fault_strength_validation,
)


HYPOTHESES = (
    "H1",
    "H3",
    "H4",
    "H5",
    "H8",
    "H9",
)

EXPECTED_STRENGTHS = {
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
}


def test_fault_strength_validation_returns_expected_number_of_observations():
    result = run_fault_strength_validation()

    # Six hypotheses × five strengths.
    assert len(result.observations) == 30


def test_all_expected_hypotheses_are_present():
    result = run_fault_strength_validation()

    hypotheses = {
        observation.hypothesis
        for observation in result.observations
    }

    assert hypotheses == set(HYPOTHESES)


def test_each_hypothesis_has_five_strength_observations():
    result = run_fault_strength_validation()

    for hypothesis in HYPOTHESES:
        observations = result.for_hypothesis(hypothesis)

        assert len(observations) == 5


def test_strength_values_are_preserved():
    result = run_fault_strength_validation()

    for hypothesis in HYPOTHESES:
        observed_strengths = {
            observation.strength
            for observation in result.for_hypothesis(
                hypothesis
            )
        }

        assert observed_strengths == EXPECTED_STRENGTHS


def test_posteriors_are_valid():
    result = run_fault_strength_validation()

    for observation in result.observations:
        assert math.isfinite(
            observation.posterior
        )

        assert 0.0 <= observation.posterior <= 1.0


def test_uncertainties_are_valid():
    result = run_fault_strength_validation()

    for observation in result.observations:
        assert math.isfinite(
            observation.uncertainty
        )

        assert 0.0 <= observation.uncertainty <= 1.0


@pytest.mark.parametrize(
    "hypothesis",
    HYPOTHESES,
)
def test_evidence_changes_in_expected_direction_with_strength(
    hypothesis,
):
    result = run_fault_strength_validation()

    assert result.evidence_is_monotonic(hypothesis) is True


@pytest.mark.parametrize(
    "hypothesis",
    HYPOTHESES,
)
def test_evidence_values_are_available_for_every_strength(
    hypothesis,
):
    result = run_fault_strength_validation()

    values = result.evidence_values(hypothesis)

    assert len(values) == 5
    assert all(
        math.isfinite(value)
        for value in values
    )


def test_result_reports_all_expected_hypotheses():
    result = run_fault_strength_validation()

    assert set(result.hypotheses) == set(HYPOTHESES)


def test_fault_strength_validation_passes():
    result = run_fault_strength_validation()

    assert result.passed is True


def test_result_rejects_empty_observations():
    result = FaultStrengthValidation(
        observations=(),
    )

    assert result.observations == ()
    assert result.passed is True


def test_observation_is_immutable():
    observation = FaultStrengthObservation(
        hypothesis="H1",
        strength=0.5,
        posterior=0.8,
        uncertainty=0.2,
    )

    with pytest.raises(AttributeError):
        observation.posterior = 0.9


def test_evidence_values_are_valid():
    result = run_fault_strength_validation()

    for observation in result.observations:
        assert isinstance(
            observation.evidence_value,
            float,
        )

        assert observation.evidence_value >= 0.0


def test_evidence_direction_configuration_is_complete():
    assert set(EVIDENCE_DIRECTIONS) == set(HYPOTHESES)