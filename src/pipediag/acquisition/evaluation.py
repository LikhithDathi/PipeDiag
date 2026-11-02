from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipediag.evidence.store import (
    DatasetContextRegistry,
    TRAIN,
    VALIDATION,
    TEST,
    UNKNOWN,
)


ESTIMATOR_SCORE = "ESTIMATOR_SCORE"
EXPLICIT_METRIC = "EXPLICIT_METRIC"


@dataclass
class EvaluationEvidence:
    """
    Observed evaluation evidence.

    This class records what happened during evaluation.
    It does not determine whether the evaluation is good,
    bad, risky, or indicative of overfitting.
    """

    evidence_type: str = "evaluation"

    metric_name: str = ""
    metric_value: float | None = None

    evaluation_context: str = UNKNOWN

    feature_dataset_id: str | None = None
    target_dataset_id: str | None = None
    prediction_dataset_id: str | None = None

    sample_count: int | None = None

    source: str = UNKNOWN

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class EvaluationAcquirer:
    """
    Converts observed evaluation results into evaluation evidence.

    Dataset context is resolved conservatively from the
    DatasetContextRegistry.

    The acquirer never guesses TRAIN/VALIDATION/TEST.
    """

    def __init__(
        self,
        context_registry: DatasetContextRegistry | None = None,
    ) -> None:

        self.context_registry = (
            context_registry
            or DatasetContextRegistry()
        )

    # ------------------------------------------------------------------
    # Estimator score
    # ------------------------------------------------------------------

    def acquire_score(
        self,
        *,
        metric_name: str,
        metric_value: Any,
        feature_dataset_id: str | None = None,
        target_dataset_id: str | None = None,
        prediction_dataset_id: str | None = None,
        sample_count: int | None = None,
        evaluation_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationEvidence:

        value = self._normalize_metric_value(
            metric_value
        )

        resolved_context = (
            evaluation_context
            if evaluation_context is not None
            else self.resolve_context(
                feature_dataset_id=feature_dataset_id,
                target_dataset_id=target_dataset_id,
                prediction_dataset_id=prediction_dataset_id,
            )
        )

        return EvaluationEvidence(
            metric_name=metric_name,
            metric_value=value,
            evaluation_context=resolved_context,
            feature_dataset_id=feature_dataset_id,
            target_dataset_id=target_dataset_id,
            prediction_dataset_id=prediction_dataset_id,
            sample_count=sample_count,
            source=ESTIMATOR_SCORE,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Context resolution
    # ------------------------------------------------------------------

    def resolve_context(
        self,
        *,
        feature_dataset_id: str | None = None,
        target_dataset_id: str | None = None,
        prediction_dataset_id: str | None = None,
    ) -> str:

        dataset_ids = [
            dataset_id
            for dataset_id in (
                feature_dataset_id,
                target_dataset_id,
                prediction_dataset_id,
            )
            if dataset_id is not None
        ]

        if not dataset_ids:
            return UNKNOWN

        contexts = [
            self.context_registry.get_context(
                dataset_id
            )
            for dataset_id in dataset_ids
        ]

        known_contexts = {
            context
            for context in contexts
            if context != UNKNOWN
        }

        # If nothing is known, the evaluation context
        # cannot be determined.
        if not known_contexts:
            return UNKNOWN

        # Every known dataset must agree.
        if len(known_contexts) != 1:
            return UNKNOWN

        resolved_context = next(
            iter(known_contexts)
        )

        # If any dataset has an explicit UNKNOWN context
        # while others are known, remain conservative.
        if UNKNOWN in contexts:
            return UNKNOWN

        if resolved_context not in {
            TRAIN,
            VALIDATION,
            TEST,
        }:
            return UNKNOWN

        return resolved_context

    # ------------------------------------------------------------------
    # Metric normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_metric_value(
        value: Any,
    ) -> float | None:

        try:
            numeric_value = float(value)

        except (TypeError, ValueError):
            return None

        if not (
            numeric_value == numeric_value
        ):
            return None

        return numeric_value