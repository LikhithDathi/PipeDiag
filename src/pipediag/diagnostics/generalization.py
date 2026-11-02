from __future__ import annotations

from dataclasses import dataclass

from pipediag.acquisition.evaluation import EvaluationEvidence
from pipediag.evidence.store import (
    TRAIN,
    TEST,
)


HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
LOWER_IS_BETTER = "LOWER_IS_BETTER"
UNKNOWN_DIRECTION = "UNKNOWN"


COMPUTED = "COMPUTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass
class GeneralizationEvidence:
    """
    Derived evidence describing the observed difference
    between training and test evaluation performance.

    This class records the result of the generalization
    comparison. It does not determine whether the model
    is overfitting.
    """

    evidence_type: str = "generalization"

    metric_name: str = ""

    train_value: float | None = None
    test_value: float | None = None

    metric_direction: str = UNKNOWN_DIRECTION

    gap: float | None = None

    status: str = INSUFFICIENT_EVIDENCE

    reason: str = ""


class GeneralizationDiagnostic:
    """
    Computes a direction-normalized generalization gap
    from observed evaluation evidence.

    The diagnostic establishes the observed relationship
    between training and test performance.

    It does not determine whether the model is overfitting.
    """

    def analyze(
        self,
        evaluations: list[EvaluationEvidence],
        metric_direction: str,
    ) -> GeneralizationEvidence:

        # --------------------------------------------------------------
        # Validate metric direction
        # --------------------------------------------------------------

        if metric_direction not in {
            HIGHER_IS_BETTER,
            LOWER_IS_BETTER,
        }:
            return GeneralizationEvidence(
                metric_direction=metric_direction,
                status=INSUFFICIENT_EVIDENCE,
                reason="Metric direction is unavailable or unknown.",
            )

        # --------------------------------------------------------------
        # Find TRAIN and TEST evaluations
        # --------------------------------------------------------------

        train_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.evaluation_context == TRAIN
        ]

        test_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.evaluation_context == TEST
        ]

        # --------------------------------------------------------------
        # Both contexts are required.
        #
        # Validation is intentionally not substituted for TEST.
        # UNKNOWN is intentionally ignored.
        # --------------------------------------------------------------

        if not train_evaluations:
            return GeneralizationEvidence(
                metric_direction=metric_direction,
                status=INSUFFICIENT_EVIDENCE,
                reason="No TRAIN evaluation evidence is available.",
            )

        if not test_evaluations:
            return GeneralizationEvidence(
                metric_direction=metric_direction,
                status=INSUFFICIENT_EVIDENCE,
                reason="No TEST evaluation evidence is available.",
            )

        # --------------------------------------------------------------
        # Match TRAIN and TEST evaluations by metric name.
        # --------------------------------------------------------------

        train_metrics = {
            evaluation.metric_name: evaluation
            for evaluation in train_evaluations
        }

        test_metrics = {
            evaluation.metric_name: evaluation
            for evaluation in test_evaluations
        }

        common_metrics = set(train_metrics) & set(test_metrics)

        if not common_metrics:
            return GeneralizationEvidence(
                metric_direction=metric_direction,
                status=NOT_COMPARABLE,
                reason=(
                    "TRAIN and TEST evaluations do not contain "
                    "the same metric."
                ),
            )

        # --------------------------------------------------------------
        # One metric comparison is required for one diagnostic result.
        #
        # If multiple common metrics exist, use the first deterministic
        # metric name rather than silently combining different metrics.
        # --------------------------------------------------------------

        metric_name = sorted(common_metrics)[0]

        train_evaluation = train_metrics[metric_name]
        test_evaluation = test_metrics[metric_name]

        # --------------------------------------------------------------
        # Metric values must actually be available.
        # --------------------------------------------------------------

        if (
            train_evaluation.metric_value is None
            or test_evaluation.metric_value is None
        ):
            return GeneralizationEvidence(
                metric_name=metric_name,
                metric_direction=metric_direction,
                train_value=train_evaluation.metric_value,
                test_value=test_evaluation.metric_value,
                status=INSUFFICIENT_EVIDENCE,
                reason="TRAIN or TEST metric value is unavailable.",
            )

        train_value = float(train_evaluation.metric_value)
        test_value = float(test_evaluation.metric_value)

        # --------------------------------------------------------------
        # Direction-normalized gap
        #
        # Higher is better:
        #
        #     G = train - test
        #
        # Lower is better:
        #
        #     G = test - train
        #
        # Therefore:
        #
        #     G > 0  -> test performance is worse
        #     G = 0  -> equal performance
        #     G < 0  -> test performance is better
        # --------------------------------------------------------------

        if metric_direction == HIGHER_IS_BETTER:
            gap = train_value - test_value
        else:
            gap = test_value - train_value

        gap = round(gap, 10)

        return GeneralizationEvidence(
            metric_name=metric_name,
            train_value=train_value,
            test_value=test_value,
            metric_direction=metric_direction,
            gap=gap,
            status=COMPUTED,
            reason="TRAIN and TEST evaluations were directly comparable.",
        )