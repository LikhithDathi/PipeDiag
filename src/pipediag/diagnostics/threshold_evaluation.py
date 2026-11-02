from __future__ import annotations

from dataclasses import dataclass

from pipediag.diagnostics.scenario_result import ScenarioResult


@dataclass(frozen=True)
class ThresholdEvaluation:
    threshold: float
    total_scenarios: int
    reported_scenarios: int
    abstained_scenarios: int
    correct_reports: int
    incorrect_reports: int

    @property
    def coverage(self) -> float:
        if self.total_scenarios == 0:
            return 0.0

        return self.reported_scenarios / self.total_scenarios

    @property
    def abstention_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0

        return self.abstained_scenarios / self.total_scenarios

    @property
    def selective_accuracy(self) -> float:
        if self.reported_scenarios == 0:
            return 0.0

        return self.correct_reports / self.reported_scenarios

    @property
    def false_diagnosis_rate(self) -> float:
        if self.reported_scenarios == 0:
            return 0.0

        return self.incorrect_reports / self.reported_scenarios


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )


def _validate_result(result: ScenarioResult) -> None:
    if result.expected_hypothesis is None:
        raise ValueError(
            "threshold evaluation requires scenarios "
            "with an expected hypothesis"
        )


def _reported_hypothesis(
    result: ScenarioResult,
    threshold: float,
) -> str | None:
    candidates = [
        (hypothesis, probability)
        for hypothesis, probability in result.posterior.items()
        if probability >= threshold
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item[1],
    )[0]


def evaluate_threshold(
    results: list[ScenarioResult],
    *,
    threshold: float,
) -> ThresholdEvaluation:
    _validate_threshold(threshold)

    total_scenarios = len(results)

    reported_scenarios = 0
    abstained_scenarios = 0
    correct_reports = 0
    incorrect_reports = 0

    for result in results:
        _validate_result(result)

        reported_hypothesis = _reported_hypothesis(
            result,
            threshold,
        )

        if reported_hypothesis is None:
            abstained_scenarios += 1
            continue

        reported_scenarios += 1

        if reported_hypothesis == result.expected_hypothesis:
            correct_reports += 1
        else:
            incorrect_reports += 1

    return ThresholdEvaluation(
        threshold=threshold,
        total_scenarios=total_scenarios,
        reported_scenarios=reported_scenarios,
        abstained_scenarios=abstained_scenarios,
        correct_reports=correct_reports,
        incorrect_reports=incorrect_reports,
    )


def evaluate_thresholds(
    results: list[ScenarioResult],
    *,
    thresholds: tuple[float, ...] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ),
) -> tuple[ThresholdEvaluation, ...]:
    if not thresholds:
        raise ValueError(
            "thresholds cannot be empty"
        )

    for threshold in thresholds:
        _validate_threshold(threshold)

    return tuple(
        evaluate_threshold(
            results,
            threshold=threshold,
        )
        for threshold in thresholds
    )