import math

import pytest

from pipediag.diagnostics.evaluation import (
    DiagnosticEvaluation,
    evaluate_diagnostic_results,
)
from pipediag.diagnostics.scenario_result import ScenarioResult


def _result(
    scenario_id,
    expected,
    posterior,
    uncertainty=0.2,
):
    return ScenarioResult(
        scenario_id=scenario_id,
        expected_hypothesis=expected,
        posterior=posterior,
        uncertainty=uncertainty,
    )


def _posterior(**values):
    return values


def test_perfect_predictions_have_perfect_top1_accuracy():
    results = [
        _result(
            "s1",
            "H1",
            _posterior(
                H1=0.9,
                H2=0.1,
            ),
        ),
        _result(
            "s2",
            "H2",
            _posterior(
                H1=0.1,
                H2=0.9,
            ),
        ),
    ]

    evaluation = evaluate_diagnostic_results(results)

    assert evaluation.top1_accuracy == 1.0


def test_top2_coverage_is_computed():
    results = [
        _result(
            "s1",
            "H1",
            _posterior(
                H2=0.6,
                H1=0.3,
                H3=0.1,
            ),
        ),
        _result(
            "s2",
            "H3",
            _posterior(
                H1=0.5,
                H3=0.3,
                H2=0.2,
            ),
        ),
    ]

    evaluation = evaluate_diagnostic_results(results)

    assert evaluation.top1_accuracy == 0.0
    assert evaluation.top2_coverage == 1.0


def test_top3_coverage_is_computed():
    results = [
        _result(
            "s1",
            "H3",
            _posterior(
                H1=0.5,
                H2=0.3,
                H3=0.2,
            ),
        ),
    ]

    evaluation = evaluate_diagnostic_results(results)

    assert evaluation.top1_accuracy == 0.0
    assert evaluation.top2_coverage == 0.0
    assert evaluation.top3_coverage == 1.0


def test_brier_score_is_zero_for_certain_correct_prediction():
    results = [
        _result(
            "s1",
            "H1",
            _posterior(
                H1=1.0,
                H2=0.0,
            ),
        ),
    ]

    evaluation = evaluate_diagnostic_results(results)

    assert math.isclose(
        evaluation.brier_score,
        0.0,
    )


def test_brier_score_matches_multiclass_definition():
    results = [
        _result(
            "s1",
            "H1",
            _posterior(
                H1=0.7,
                H2=0.2,
                H3=0.1,
            ),
        ),
    ]

    evaluation = evaluate_diagnostic_results(results)

    expected = (
        (0.7 - 1.0) ** 2
        + (0.2 - 0.0) ** 2
        + (0.1 - 0.0) ** 2
    )

    assert math.isclose(
        evaluation.brier_score,
        expected,
    )


def test_mean_uncertainty_is_computed():
    results = [
        _result(
            "s1",
            "H1",
            _posterior(H1=0.8, H2=0.2),
            uncertainty=0.2,
        ),
        _result(
            "s2",
            "H2",
            _posterior(H1=0.2, H2=0.8),
            uncertainty=0.6,
        ),
    ]

    evaluation = evaluate_diagnostic_results(results)

    assert math.isclose(
        evaluation.mean_uncertainty,
        0.4,
    )


def test_clean_negative_control_is_excluded_from_supervised_metrics():
    results = [
        _result(
            "fault",
            "H1",
            _posterior(H1=0.9, H2=0.1),
        ),
        _result(
            "clean",
            None,
            _posterior(H1=0.2, H2=0.8),
            uncertainty=0.8,
        ),
    ]

    evaluation = evaluate_diagnostic_results(results)

    assert evaluation.total_scenarios == 2
    assert evaluation.evaluated_scenarios == 1
    assert evaluation.top1_accuracy == 1.0


def test_ece_reflects_confidence_accuracy_gap():
    results = [
        _result(
            "s1",
            "H1",
            _posterior(H1=0.8, H2=0.2),
        ),
        _result(
            "s2",
            "H2",
            _posterior(H1=0.2, H2=0.8),
        ),
    ]

    evaluation = evaluate_diagnostic_results(
        results,
        calibration_bins=10,
    )

    assert math.isclose(
        evaluation.expected_calibration_error,
        0.2,
    )


def test_evaluation_rejects_empty_supervised_dataset():
    with pytest.raises(ValueError, match="expected hypothesis"):
        evaluate_diagnostic_results(
            [
                _result(
                    "clean",
                    None,
                    _posterior(H1=0.5, H2=0.5),
                )
            ]
        )


def test_evaluation_rejects_invalid_posterior():
    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        evaluate_diagnostic_results(
            [
                _result(
                    "s1",
                    "H1",
                    _posterior(
                        H1=0.8,
                        H2=0.8,
                    ),
                )
            ]
        )


def test_evaluation_result_validates_metrics():
    result = DiagnosticEvaluation(
        total_scenarios=2,
        evaluated_scenarios=2,
        top1_accuracy=1.0,
        top2_coverage=1.0,
        top3_coverage=1.0,
        brier_score=0.0,
        expected_calibration_error=0.0,
        mean_uncertainty=0.0,
    )

    assert result.evaluated_scenarios == 2


def test_evaluation_result_rejects_invalid_accuracy():
    with pytest.raises(ValueError, match="between 0 and 1"):
        DiagnosticEvaluation(
            total_scenarios=1,
            evaluated_scenarios=1,
            top1_accuracy=1.5,
            top2_coverage=1.0,
            top3_coverage=1.0,
            brier_score=0.0,
            expected_calibration_error=0.0,
            mean_uncertainty=0.0,
        )