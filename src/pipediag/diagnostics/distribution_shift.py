from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipediag.evidence.store import UNKNOWN


SHIFT_DETECTED = "SHIFT_DETECTED"
NO_SHIFT = "NO_SHIFT"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class DistributionShiftEvidence:
    """
    Observed feature-distribution comparison evidence.

    This class records the measured difference between two
    observed datasets.

    It does not determine whether distribution shift caused
    a model-performance problem.
    """

    evidence_type: str = "distribution_shift"

    reference_dataset_id: str | None = None
    comparison_dataset_id: str | None = None

    reference_context: str = UNKNOWN
    comparison_context: str = UNKNOWN

    feature_shifts: dict[str, float] = field(
        default_factory=dict
    )

    max_shift: float | None = None

    shift_threshold: float = 0.2

    status: str = INSUFFICIENT_EVIDENCE
    reason: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class DistributionShiftDiagnostic:
    """
    Compares feature distributions between two datasets.

    The diagnostic uses a simple empirical CDF distance
    (maximum absolute difference between the two empirical
    cumulative distributions) for each common feature.

    It reports observed distribution differences only.

    It does not make causal claims about:
        - model performance
        - overfitting
        - leakage
        - model failure
    """

    def __init__(
        self,
        shift_threshold: float = 0.2,
    ) -> None:

        if shift_threshold < 0:
            raise ValueError(
                "shift_threshold must be greater than or equal to zero."
            )

        self.shift_threshold = float(
            shift_threshold
        )

    def analyze(
        self,
        *,
        reference_evidence: dict[str, Any] | None,
        comparison_evidence: dict[str, Any] | None,
    ) -> DistributionShiftEvidence:

        if not reference_evidence:
            return DistributionShiftEvidence(
                shift_threshold=self.shift_threshold,
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Reference distribution evidence "
                    "was not available."
                ),
            )

        if not comparison_evidence:
            return DistributionShiftEvidence(
                shift_threshold=self.shift_threshold,
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Comparison distribution evidence "
                    "was not available."
                ),
            )

        reference_dataset_id = (
            reference_evidence.get("dataset_id")
        )

        comparison_dataset_id = (
            comparison_evidence.get("dataset_id")
        )

        reference_context = (
            reference_evidence.get(
                "context",
                UNKNOWN,
            )
        )

        comparison_context = (
            comparison_evidence.get(
                "context",
                UNKNOWN,
            )
        )

        reference_distributions = (
            reference_evidence.get(
                "distributions"
            )
        )

        comparison_distributions = (
            comparison_evidence.get(
                "distributions"
            )
        )

        if not reference_distributions:
            return DistributionShiftEvidence(
                reference_dataset_id=reference_dataset_id,
                comparison_dataset_id=comparison_dataset_id,
                reference_context=reference_context,
                comparison_context=comparison_context,
                shift_threshold=self.shift_threshold,
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Reference feature distributions "
                    "were not available."
                ),
            )

        if not comparison_distributions:
            return DistributionShiftEvidence(
                reference_dataset_id=reference_dataset_id,
                comparison_dataset_id=comparison_dataset_id,
                reference_context=reference_context,
                comparison_context=comparison_context,
                shift_threshold=self.shift_threshold,
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Comparison feature distributions "
                    "were not available."
                ),
            )

        if not isinstance(
            reference_distributions,
            dict,
        ):
            return DistributionShiftEvidence(
                reference_dataset_id=reference_dataset_id,
                comparison_dataset_id=comparison_dataset_id,
                reference_context=reference_context,
                comparison_context=comparison_context,
                shift_threshold=self.shift_threshold,
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Reference feature distributions "
                    "must be a mapping."
                ),
            )

        if not isinstance(
            comparison_distributions,
            dict,
        ):
            return DistributionShiftEvidence(
                reference_dataset_id=reference_dataset_id,
                comparison_dataset_id=comparison_dataset_id,
                reference_context=reference_context,
                comparison_context=comparison_context,
                shift_threshold=self.shift_threshold,
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "Comparison feature distributions "
                    "must be a mapping."
                ),
            )

        common_features = (
            set(reference_distributions)
            & set(comparison_distributions)
        )

        if not common_features:
            return DistributionShiftEvidence(
                reference_dataset_id=reference_dataset_id,
                comparison_dataset_id=comparison_dataset_id,
                reference_context=reference_context,
                comparison_context=comparison_context,
                shift_threshold=self.shift_threshold,
                status=INSUFFICIENT_EVIDENCE,
                reason=(
                    "No common features were available "
                    "for comparison."
                ),
            )

        feature_shifts: dict[str, float] = {}

        for feature in common_features:

            reference_values = (
                reference_distributions[feature]
            )

            comparison_values = (
                comparison_distributions[feature]
            )

            reference_values = (
                self._normalize_values(
                    reference_values
                )
            )

            comparison_values = (
                self._normalize_values(
                    comparison_values
                )
            )

            if (
                reference_values is None
                or comparison_values is None
            ):
                return DistributionShiftEvidence(
                    reference_dataset_id=reference_dataset_id,
                    comparison_dataset_id=comparison_dataset_id,
                    reference_context=reference_context,
                    comparison_context=comparison_context,
                    feature_shifts={},
                    max_shift=None,
                    shift_threshold=self.shift_threshold,
                    status=INSUFFICIENT_EVIDENCE,
                    reason=(
                        "One or more feature distributions "
                        "contained invalid values."
                    ),
                )

            if (
                not reference_values
                or not comparison_values
            ):
                return DistributionShiftEvidence(
                    reference_dataset_id=reference_dataset_id,
                    comparison_dataset_id=comparison_dataset_id,
                    reference_context=reference_context,
                    comparison_context=comparison_context,
                    feature_shifts={},
                    max_shift=None,
                    shift_threshold=self.shift_threshold,
                    status=INSUFFICIENT_EVIDENCE,
                    reason=(
                        "One or more feature distributions "
                        "contained no observations."
                    ),
                )

            feature_shifts[feature] = (
                self._wasserstein_distance(
                    reference_values,
                    comparison_values,
                )
            )

        max_shift = max(
            feature_shifts.values()
        )

        if max_shift > self.shift_threshold:
            status = SHIFT_DETECTED

            reason = (
                "At least one common feature shows an "
                "observed distribution difference above "
                "the configured threshold."
            )

        else:
            status = NO_SHIFT

            reason = (
                "Observed feature-distribution differences "
                "are within the configured threshold."
            )

        return DistributionShiftEvidence(
            reference_dataset_id=reference_dataset_id,
            comparison_dataset_id=comparison_dataset_id,
            reference_context=reference_context,
            comparison_context=comparison_context,
            feature_shifts=feature_shifts,
            max_shift=max_shift,
            shift_threshold=self.shift_threshold,
            status=status,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Value normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_values(
        values: Any,
    ) -> list[float] | None:

        try:
            iterator = iter(values)

        except TypeError:
            return None

        normalized: list[float] = []

        for value in iterator:

            try:
                numeric_value = float(value)

            except (TypeError, ValueError):
                return None

            if numeric_value != numeric_value:
                return None

            normalized.append(
                numeric_value
            )

        return normalized

    # ------------------------------------------------------------------
    # Empirical distribution distance
    # ------------------------------------------------------------------

    @staticmethod
    def _wasserstein_distance(
        reference_values: list[float],
        comparison_values: list[float],
    ) -> float:

        reference_sorted = sorted(reference_values)
        comparison_sorted = sorted(comparison_values)

        reference_count = len(reference_sorted)
        comparison_count = len(comparison_sorted)

        if reference_count == comparison_count:
            return sum(
                abs(
                    reference_value
                    - comparison_value
                )
                for reference_value, comparison_value
                in zip(
                    reference_sorted,
                    comparison_sorted,
                )
            ) / reference_count

        # For unequal sample sizes, compare the empirical
        # quantiles at a common set of probabilities.
        count = max(
            reference_count,
            comparison_count,
        )

        total_distance = 0.0

        for index in range(count):

            probability = (
                index / (count - 1)
                if count > 1
                else 0.0
            )

            reference_index = min(
                int(
                    probability
                    * (reference_count - 1)
                ),
                reference_count - 1,
            )

            comparison_index = min(
                int(
                    probability
                    * (comparison_count - 1)
                ),
                comparison_count - 1,
            )

            total_distance += abs(
                reference_sorted[reference_index]
                - comparison_sorted[comparison_index]
            )

        return total_distance / count

    @staticmethod
    def _cdf_value(
        sorted_values: list[float],
        value: float,
    ) -> int:

        count = 0

        for item in sorted_values:

            if item <= value:
                count += 1

            else:
                break

        return count