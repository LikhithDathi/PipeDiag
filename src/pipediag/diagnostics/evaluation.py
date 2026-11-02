from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from pipediag.diagnostics.scenario_result import ScenarioResult


@dataclass(frozen=True)
class DiagnosticEvaluation:
    total_scenarios: int
    evaluated_scenarios: int
    top1_accuracy: float
    top2_coverage: float
    top3_coverage: float
    brier_score: float
    expected_calibration_error: float
    mean_uncertainty: float

    def __post_init__(self) -> None:
        if self.total_scenarios < 0:
            raise ValueError("total_scenarios must be non-negative")

        if self.evaluated_scenarios < 0:
            raise ValueError("evaluated_scenarios must be non-negative")

        if self.evaluated_scenarios > self.total_scenarios:
            raise ValueError(
                "evaluated_scenarios cannot exceed total_scenarios"
            )

        for name, value in (
            ("top1_accuracy", self.top1_accuracy),
            ("top2_coverage", self.top2_coverage),
            ("top3_coverage", self.top3_coverage),
            ("brier_score", self.brier_score),
            (
                "expected_calibration_error",
                self.expected_calibration_error,
            ),
            ("mean_uncertainty", self.mean_uncertainty),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        for name, value in (
            ("top1_accuracy", self.top1_accuracy),
            ("top2_coverage", self.top2_coverage),
            ("top3_coverage", self.top3_coverage),
            ("expected_calibration_error", self.expected_calibration_error),
            ("mean_uncertainty", self.mean_uncertainty),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        if self.brier_score < 0.0:
            raise ValueError("brier_score must be non-negative")


def _validate_posterior(
    posterior: dict[str, float],
) -> None:
    if not posterior:
        raise ValueError("posterior cannot be empty")

    total = 0.0

    for hypothesis, probability in posterior.items():
        if not hypothesis:
            raise ValueError(
                "posterior hypothesis IDs must be non-empty"
            )

        if not isfinite(probability):
            raise ValueError(
                "posterior probabilities must be finite"
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "posterior probabilities must be between 0 and 1"
            )

        total += probability

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "posterior probabilities must sum to 1"
        )


def _ranked_hypotheses(
    posterior: dict[str, float],
) -> list[str]:
    return sorted(
        posterior,
        key=posterior.get,
        reverse=True,
    )


def _brier_score(
    posterior: dict[str, float],
    expected_hypothesis: str,
) -> float:
    if expected_hypothesis not in posterior:
        raise ValueError(
            "expected hypothesis must exist in posterior"
        )

    return sum(
        (
            probability
            - (1.0 if hypothesis == expected_hypothesis else 0.0)
        )
        ** 2
        for hypothesis, probability in posterior.items()
    )


def _expected_calibration_error(
    results: list[ScenarioResult],
    bins: int = 10,
) -> float:
    if not results:
        return 0.0

    if bins <= 0:
        raise ValueError("bins must be positive")

    bin_counts = [0] * bins
    bin_confidence = [0.0] * bins
    bin_correct = [0] * bins

    for result in results:
        if result.expected_hypothesis is None:
            continue

        posterior = result.posterior
        _validate_posterior(posterior)

        predicted = result.predicted_hypothesis
        confidence = posterior[predicted]
        correct = predicted == result.expected_hypothesis

        bin_index = min(
            int(confidence * bins),
            bins - 1,
        )

        bin_counts[bin_index] += 1
        bin_confidence[bin_index] += confidence

        if correct:
            bin_correct[bin_index] += 1

    evaluated = sum(bin_counts)

    if evaluated == 0:
        return 0.0

    error = 0.0

    for count, confidence_sum, correct_count in zip(
        bin_counts,
        bin_confidence,
        bin_correct,
    ):
        if count == 0:
            continue

        average_confidence = confidence_sum / count
        accuracy = correct_count / count

        error += (
            count / evaluated
        ) * abs(accuracy - average_confidence)

    return error


def evaluate_diagnostic_results(
    results: list[ScenarioResult],
    *,
    calibration_bins: int = 10,
) -> DiagnosticEvaluation:
    if results is None:
        raise ValueError("results cannot be None")

    if calibration_bins <= 0:
        raise ValueError(
            "calibration_bins must be positive"
        )

    evaluated_results = [
        result
        for result in results
        if result.expected_hypothesis is not None
    ]

    if not evaluated_results:
        raise ValueError(
            "at least one scenario with an expected hypothesis "
            "is required"
        )

    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    total_brier = 0.0
    total_uncertainty = 0.0

    for result in evaluated_results:
        posterior = result.posterior

        _validate_posterior(posterior)

        expected = result.expected_hypothesis

        if expected not in posterior:
            raise ValueError(
                "expected hypothesis must exist in posterior"
            )

        ranked = _ranked_hypotheses(posterior)

        if ranked[0] == expected:
            top1_correct += 1

        if expected in ranked[:2]:
            top2_correct += 1

        if expected in ranked[:3]:
            top3_correct += 1

        total_brier += _brier_score(
            posterior,
            expected,
        )

        if not isfinite(result.uncertainty):
            raise ValueError(
                "scenario uncertainty must be finite"
            )

        if not 0.0 <= result.uncertainty <= 1.0:
            raise ValueError(
                "scenario uncertainty must be between 0 and 1"
            )

        total_uncertainty += result.uncertainty

    count = len(evaluated_results)

    return DiagnosticEvaluation(
        total_scenarios=len(results),
        evaluated_scenarios=count,
        top1_accuracy=top1_correct / count,
        top2_coverage=top2_correct / count,
        top3_coverage=top3_correct / count,
        brier_score=total_brier / count,
        expected_calibration_error=_expected_calibration_error(
            evaluated_results,
            bins=calibration_bins,
        ),
        mean_uncertainty=total_uncertainty / count,
    )