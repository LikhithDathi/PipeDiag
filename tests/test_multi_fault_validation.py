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


def test_multi_fault_returns_complete_posterior():
    validation = (
        run_distribution_shift_imbalance_validation()
    )

    assert set(validation.posterior) == set(
        HYPOTHESIS_IDS
    )


def test_multi_fault_posterior_is_normalized():
    validation = (
        run_distribution_shift_imbalance_validation()
    )

    assert sum(validation.posterior.values()) == pytest.approx(
        1.0
    )


def test_multi_fault_posterior_probabilities_are_valid():
    validation = (
        run_distribution_shift_imbalance_validation()
    )

    for probability in validation.posterior.values():
        assert 0.0 <= probability <= 1.0


def test_multi_fault_uncertainty_is_valid():
    validation = (
        run_distribution_shift_imbalance_validation()
    )

    assert 0.0 <= validation.uncertainty <= 1.0


def test_multi_fault_has_no_single_fault_ground_truth():
    validation = (
        run_distribution_shift_imbalance_validation()
    )

    assert validation.result.expected_hypothesis is None


def test_multi_fault_prediction_is_known_hypothesis():
    validation = (
        run_distribution_shift_imbalance_validation()
    )

    assert validation.predicted_hypothesis in HYPOTHESIS_IDS




def test_multi_fault_retains_strong_support_for_dominant_h3():
    validation = (
        run_distribution_shift_imbalance_validation()
    )

    assert validation.posterior["H3"] > 0.25
