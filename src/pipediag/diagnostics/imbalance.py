from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipediag.evidence.store import UNKNOWN


IMBALANCED = "IMBALANCED"
NO_IMBALANCE = "NO_IMBALANCE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class ImbalanceEvidence:
    """
    Observed target-distribution evidence.

    This diagnostic describes the observed class distribution.
    It does not determine whether imbalance caused a model
    performance problem.
    """

    evidence_type: str = "imbalance"

    dataset_id: str | None = None
    context: str = UNKNOWN

    class_count: int | None = None
    class_distribution: dict[Any, float] = field(
        default_factory=dict
    )

    majority_class: Any = None
    minority_class: Any = None

    majority_proportion: float | None = None
    minority_proportion: float | None = None

    imbalance_ratio: float | None = None
    imbalance_ratio_threshold: float = 3.0

    status: str = INSUFFICIENT_EVIDENCE
    reason: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class ImbalanceDiagnostic:
    """
    Analyzes observed target-class distributions.

    The diagnostic identifies whether the observed distribution
    crosses a configured imbalance-ratio threshold.

    It does not make causal claims about model performance,
    generalization, or model quality.
    """

    def __init__(
        self,
        imbalance_ratio_threshold: float = 3.0,
    ) -> None:

        if imbalance_ratio_threshold <= 0:
            raise ValueError(
                "imbalance_ratio_threshold must be greater than zero."
            )

        self.imbalance_ratio_threshold = (
            float(imbalance_ratio_threshold)
        )

    def analyze(
        self,
        *,
        target_evidence: dict[str, Any] | None,
    ) -> ImbalanceEvidence:

        if not target_evidence:
            return ImbalanceEvidence(
                imbalance_ratio_threshold=(
                    self.imbalance_ratio_threshold
                ),
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Target-distribution evidence was not available."
                ),
            )

        dataset_id = target_evidence.get(
            "dataset_id"
        )

        context = target_evidence.get(
            "context",
            UNKNOWN,
        )

        class_distribution = target_evidence.get(
            "class_distribution"
        )

        if not class_distribution:
            return ImbalanceEvidence(
                dataset_id=dataset_id,
                context=context,
                imbalance_ratio_threshold=(
                    self.imbalance_ratio_threshold
                ),
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Target class distribution was not available."
                ),
            )

        if not isinstance(
            class_distribution,
            dict,
        ):
            return ImbalanceEvidence(
                dataset_id=dataset_id,
                context=context,
                imbalance_ratio_threshold=(
                    self.imbalance_ratio_threshold
                ),
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Target class distribution must be a mapping."
                ),
            )

        normalized_distribution = (
            self._normalize_distribution(
                class_distribution
            )
        )

        if normalized_distribution is None:
            return ImbalanceEvidence(
                dataset_id=dataset_id,
                context=context,
                imbalance_ratio_threshold=(
                    self.imbalance_ratio_threshold
                ),
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Target class distribution contains "
                    "invalid proportions."
                ),
            )

        class_count = len(
            normalized_distribution
        )

        if class_count < 2:
            return ImbalanceEvidence(
                dataset_id=dataset_id,
                context=context,
                class_count=class_count,
                class_distribution=normalized_distribution,
                imbalance_ratio_threshold=(
                    self.imbalance_ratio_threshold
                ),
                status=NOT_APPLICABLE,
                reason=(
                    "Class imbalance requires at least "
                    "two target classes."
                ),
            )

        majority_class = max(
            normalized_distribution,
            key=normalized_distribution.get,
        )

        minority_class = min(
            normalized_distribution,
            key=normalized_distribution.get,
        )

        majority_proportion = (
            normalized_distribution[
                majority_class
            ]
        )

        minority_proportion = (
            normalized_distribution[
                minority_class
            ]
        )

        if minority_proportion <= 0:
            return ImbalanceEvidence(
                dataset_id=dataset_id,
                context=context,
                class_count=class_count,
                class_distribution=normalized_distribution,
                majority_class=majority_class,
                minority_class=minority_class,
                majority_proportion=majority_proportion,
                minority_proportion=minority_proportion,
                imbalance_ratio=None,
                imbalance_ratio_threshold=(
                    self.imbalance_ratio_threshold
                ),
                status=IMBALANCED,
                reason=(
                    "The minority class has zero observed "
                    "proportion."
                ),
            )

        imbalance_ratio = (
            majority_proportion
            / minority_proportion
        )

        if (
            imbalance_ratio
            >= self.imbalance_ratio_threshold
        ):
            status = IMBALANCED

            reason = (
                "The observed majority-to-minority "
                "class ratio meets or exceeds the "
                "configured imbalance threshold."
            )

        else:
            status = NO_IMBALANCE

            reason = (
                "The observed majority-to-minority "
                "class ratio is below the configured "
                "imbalance threshold."
            )

        return ImbalanceEvidence(
            dataset_id=dataset_id,
            context=context,
            class_count=class_count,
            class_distribution=normalized_distribution,
            majority_class=majority_class,
            minority_class=minority_class,
            majority_proportion=majority_proportion,
            minority_proportion=minority_proportion,
            imbalance_ratio=imbalance_ratio,
            imbalance_ratio_threshold=(
                self.imbalance_ratio_threshold
            ),
            status=status,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Distribution normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_distribution(
        distribution: dict[Any, Any],
    ) -> dict[Any, float] | None:

        normalized: dict[Any, float] = {}

        for class_label, proportion in distribution.items():

            try:
                value = float(proportion)

            except (TypeError, ValueError):
                return None

            if value < 0:
                return None

            if value != value:
                return None

            normalized[class_label] = value

        total = sum(
            normalized.values()
        )

        if total <= 0:
            return None

        # Normalize so callers do not have to provide perfectly
        # normalized proportions.
        return {
            class_label: value / total
            for class_label, value
            in normalized.items()
        }