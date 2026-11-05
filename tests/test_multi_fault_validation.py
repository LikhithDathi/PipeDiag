
import pytest

from experiments.multi_fault_validation import (
    HYPOTHESIS_IDS,
    build_distribution_shift_imbalance_scenario,
    build_multi_fault_engine,
    run_distribution_shift_imbalance_validation,
)


def test_multi_fault_scenario_is_configured_correctly():
    scenario = build_distribution_shift_imbalance_scenario()

    assert scenario.scenario_id == "multi_fault_H3_H4"
    assert scenario.expected_hypothesis is None
    assert scenario.metadata["scenario_type"] == "multi_fault"
    assert scenario.metadata["faults_injected"] == (
        "H3",
        "H4",
    )


def test_multi_fault_engine_can_be_built():
    engine = build_multi_fault_engine()

    assert engine is not None


def test_multi_fault_returns_only_evidence_supported_hypotheses():
    validation = run_distribution_shift_imbalance_validation()

    assert set(validation.posterior) == {"H3", "H4"}


def test_multi_fault_posterior_is_normalized():
    validation = run_distribution_shift_imbalance_validation()

    assert sum(validation.posterior.values()) == pytest.approx(
        1.0,
        abs=1e-9,
    )


def test_multi_fault_posterior_probabilities_are_valid():
    validation = run_distribution_shift_imbalance_validation()

    for probability in validation.posterior.values():
        assert 0.0 <= probability <= 1.0


def test_multi_fault_uncertainty_is_valid():
    validation = run_distribution_shift_imbalance_validation()

    assert 0.0 <= validation.uncertainty <= 1.0

    for uncertainty in (
        validation.result.uncertainty_by_hypothesis.values()
    ):
        assert 0.0 <= uncertainty <= 1.0


def test_multi_fault_has_no_single_fault_ground_truth():
    validation = run_distribution_shift_imbalance_validation()

    assert validation.result.expected_hypothesis is None


def test_multi_fault_prediction_is_known_hypothesis():
    validation = run_distribution_shift_imbalance_validation()

    assert validation.predicted_hypothesis in {"H3", "H4"}


def test_multi_fault_retains_strong_support_for_h3():
    validation = run_distribution_shift_imbalance_validation()

    assert validation.posterior["H3"] > 0.25


def test_multi_fault_detects_both_injected_faults():
    validation = run_distribution_shift_imbalance_validation()

    # Detection uses independent Bayesian probabilities.
    independent = validation.result.independent_posterior

    assert independent["H3"] >= 0.5
    assert independent["H4"] >= 0.5

    assert set(validation.detected_hypotheses) == {
        "H3",
        "H4",
    }


def test_normalized_support_does_not_require_each_fault_above_half():
    validation = run_distribution_shift_imbalance_validation()

    assert sum(validation.posterior.values()) == pytest.approx(1.0)

    # Both faults are detected even though their normalized
    # diagnostic support is shared between them.
    assert validation.posterior["H3"] < 1.0
    assert validation.posterior["H4"] < 1.0


def test_unrelated_hypotheses_are_not_detected():
    validation = run_distribution_shift_imbalance_validation()

    unrelated = set(HYPOTHESIS_IDS) - {"H3", "H4"}

    assert not (
        set(validation.detected_hypotheses) & unrelated
    )


def test_multi_fault_result_preserves_evidence_provenance():
    validation = run_distribution_shift_imbalance_validation()

    assert {
        evidence.name
        for evidence in validation.result.evidence
    } == {"G3", "G4"}

    assert validation.result.evidence_supported_hypotheses == {
        "H3",
        "H4",
    }


def test_multi_fault_result_preserves_individual_likelihood_ratios():
    validation = run_distribution_shift_imbalance_validation()

    ratios = validation.result.evidence_log_likelihood_ratios

    assert ratios["G3"]["H3"] > 0
    assert ratios["G4"]["H4"] > 0


def test_independent_posterior_is_separate_from_normalized_posterior():
    validation = run_distribution_shift_imbalance_validation()

    normalized = validation.posterior
    independent = validation.result.independent_posterior

    assert set(normalized) == set(independent)

    assert sum(normalized.values()) == pytest.approx(1.0)

    assert sum(independent.values()) != pytest.approx(1.0)
