import math

from experiments.diagnostic_evaluation import (
    run_single_fault_diagnostic_evaluation,
)


def test_single_fault_evaluation_contains_all_scenarios():
    result = run_single_fault_diagnostic_evaluation()

    assert len(result.scenario_results) == 9
    assert result.evaluation.total_scenarios == 9
    assert result.evaluation.evaluated_scenarios == 9


def test_single_fault_top1_accuracy_is_valid():
    result = run_single_fault_diagnostic_evaluation()

    assert 0.0 <= result.evaluation.top1_accuracy <= 1.0


def test_single_fault_top2_coverage_is_valid():
    result = run_single_fault_diagnostic_evaluation()

    assert 0.0 <= result.evaluation.top2_coverage <= 1.0


def test_single_fault_top3_coverage_is_valid():
    result = run_single_fault_diagnostic_evaluation()

    assert 0.0 <= result.evaluation.top3_coverage <= 1.0


def test_single_fault_brier_score_is_valid():
    result = run_single_fault_diagnostic_evaluation()

    assert math.isfinite(
        result.evaluation.brier_score
    )

    assert result.evaluation.brier_score >= 0.0


def test_single_fault_ece_is_valid():
    result = run_single_fault_diagnostic_evaluation()

    assert math.isfinite(
        result.evaluation.expected_calibration_error
    )

    assert 0.0 <= (
        result.evaluation.expected_calibration_error
    ) <= 1.0


def test_single_fault_mean_uncertainty_is_valid():
    result = run_single_fault_diagnostic_evaluation()

    assert math.isfinite(
        result.evaluation.mean_uncertainty
    )

    assert 0.0 <= result.evaluation.mean_uncertainty <= 1.0


def test_each_single_fault_result_has_expected_hypothesis():
    result = run_single_fault_diagnostic_evaluation()

    expected = {
        f"H{i}"
        for i in range(1, 10)
    }

    observed = {
        scenario.expected_hypothesis
        for scenario in result.scenario_results
    }

    assert observed == expected


def test_each_single_fault_result_has_complete_posterior():
    result = run_single_fault_diagnostic_evaluation()

    expected = {
        f"H{i}"
        for i in range(1, 10)
    }

    for scenario_result in result.scenario_results:
        assert set(
            scenario_result.posterior
        ) == expected


def test_each_single_fault_result_has_valid_uncertainty():
    result = run_single_fault_diagnostic_evaluation()

    for scenario_result in result.scenario_results:
        assert math.isfinite(
            scenario_result.uncertainty
        )

        assert 0.0 <= scenario_result.uncertainty <= 1.0