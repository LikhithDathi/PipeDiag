import math

import pytest

from pipediag.diagnostics.diagnostic_scenario import DiagnosticScenario
from pipediag.diagnostics.scenario_result import ScenarioResult
from pipediag.diagnostics.threshold_evaluation import (
    ThresholdEvaluation,
    evaluate_threshold,
    evaluate_thresholds,
)


HYPOTHESES = (
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H8",
    "H9",
)


def _scenario(
    posterior: dict[str, float],
    expected: str = "H1",
) -> ScenarioResult:
    scenario = DiagnosticScenario(
        scenario_id="test",
        expected_hypothesis=expected,
        description="Test scenario",
    )

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        expected_hypothesis=scenario.expected_hypothesis,
        posterior=posterior,
        uncertainty=0.0,
    )


def _posterior(**values: float) -> dict[str, float]:
    remaining = 1.0 - sum(values.values())

    result = {
        hypothesis: 0.0
        for hypothesis in HYPOTHESES
    }

    result.update(values)

    if remaining > 0:
        result["H9"] += remaining

    return result


def test_threshold_evaluation_returns_expected_type():
    result = evaluate_threshold(
        [_scenario(_posterior(H1=1.0))],
        threshold=0.5,
    )

    assert isinstance(
        result,
        ThresholdEvaluation,
    )


def test_confident_correct_diagnosis_is_reported():
    result = evaluate_threshold(
        [_scenario(_posterior(H1=1.0))],
        threshold=0.5,
    )

    assert result.reported_scenarios == 1
    assert result.abstained_scenarios == 0
    assert result.correct_reports == 1
    assert result.incorrect_reports == 0


def test_inconclusive_case_is_abstained():
    posterior = {
        hypothesis: 1.0 / 9.0
        for hypothesis in HYPOTHESES
    }

    result = evaluate_threshold(
        [_scenario(posterior)],
        threshold=0.5,
    )

    assert result.reported_scenarios == 0
    assert result.abstained_scenarios == 1
    assert result.correct_reports == 0
    assert result.incorrect_reports == 0


def test_incorrect_hypothesis_above_threshold_is_counted():
    result = evaluate_threshold(
        [_scenario(
            _posterior(H2=0.8, H1=0.1),
            expected="H1",
        )],
        threshold=0.5,
    )

    assert result.reported_scenarios == 1
    assert result.correct_reports == 0
    assert result.incorrect_reports == 1


def test_multiple_hypotheses_can_cross_threshold():
    result = evaluate_threshold(
        [_scenario(
            _posterior(H1=0.45, H2=0.40),
            expected="H1",
        )],
        threshold=0.30,
    )

    assert result.reported_scenarios == 1
    assert result.correct_reports == 1
    assert result.incorrect_reports == 0


def test_coverage_is_reported_fraction():
    results = [
        _scenario(_posterior(H1=1.0)),
        _scenario(_posterior(H1=1.0)),
        _scenario(
            {
                hypothesis: 1.0 / 9.0
                for hypothesis in HYPOTHESES
            }
        ),
    ]

    evaluation = evaluate_threshold(
        results,
        threshold=0.5,
    )

    assert math.isclose(
        evaluation.coverage,
        2.0 / 3.0,
    )


def test_abstention_rate_is_reported_fraction():
    results = [
        _scenario(_posterior(H1=1.0)),
        _scenario(
            {
                hypothesis: 1.0 / 9.0
                for hypothesis in HYPOTHESES
            }
        ),
    ]

    evaluation = evaluate_threshold(
        results,
        threshold=0.5,
    )

    assert math.isclose(
        evaluation.abstention_rate,
        0.5,
    )


def test_selective_accuracy_uses_only_reported_cases():
    results = [
        _scenario(_posterior(H1=1.0)),
        _scenario(
            _posterior(H2=0.8, H1=0.1),
            expected="H1",
        ),
        _scenario(
            {
                hypothesis: 1.0 / 9.0
                for hypothesis in HYPOTHESES
            }
        ),
    ]

    evaluation = evaluate_threshold(
        results,
        threshold=0.5,
    )

    assert evaluation.correct_reports == 1
    assert evaluation.incorrect_reports == 1
    assert math.isclose(
        evaluation.selective_accuracy,
        0.5,
    )


def test_false_diagnosis_rate_uses_only_reported_cases():
    results = [
        _scenario(_posterior(H1=1.0)),
        _scenario(
            _posterior(H2=0.8, H1=0.1),
            expected="H1",
        ),
    ]

    evaluation = evaluate_threshold(
        results,
        threshold=0.5,
    )

    assert math.isclose(
        evaluation.false_diagnosis_rate,
        0.5,
    )


def test_multiple_thresholds_are_evaluated():
    results = [
        _scenario(_posterior(H1=1.0)),
    ]

    evaluations = evaluate_thresholds(
        results,
        thresholds=(0.5, 0.8),
    )

    assert len(evaluations) == 2
    assert evaluations[0].threshold == 0.5
    assert evaluations[1].threshold == 0.8


def test_threshold_must_be_between_zero_and_one():
    results = [
        _scenario(_posterior(H1=1.0)),
    ]

    with pytest.raises(ValueError):
        evaluate_threshold(
            results,
            threshold=-0.1,
        )

    with pytest.raises(ValueError):
        evaluate_threshold(
            results,
            threshold=1.1,
        )


def test_empty_threshold_list_is_rejected():
    with pytest.raises(ValueError):
        evaluate_thresholds(
            [],
            thresholds=(),
        )


def test_scenario_without_expected_hypothesis_is_rejected():
    scenario = ScenarioResult(
        scenario_id="clean",
        expected_hypothesis=None,
        posterior={
            hypothesis: 1.0 / 9.0
            for hypothesis in HYPOTHESES
        },
        uncertainty=1.0,
    )

    with pytest.raises(ValueError):
        evaluate_threshold(
            [scenario],
            threshold=0.5,
        )