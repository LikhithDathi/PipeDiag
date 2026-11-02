from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Iterable

from pipediag.diagnostics.calibration import CalibrationObservation


@dataclass(frozen=True)
class InterpretationBoundary:
    """
    Empirically derived boundaries used only for explaining evidence.

    The boundaries do not participate in Bayesian inference.

    limited:
        Evidence is within the region commonly observed under the
        non-fault calibration scenarios.

    observable:
        Evidence has moved beyond the normal calibration region but
        has not yet reached the typical fault-like region.

    substantial:
        Evidence is at or beyond the central fault-like region.
    """

    evidence_group: str
    hypothesis: str
    measurement: str

    normal_reference: float
    observable_boundary: float
    substantial_boundary: float

    direction: int

    def __post_init__(self) -> None:
        if not self.evidence_group:
            raise ValueError(
                "InterpretationBoundary requires a non-empty evidence_group"
            )

        if not self.hypothesis:
            raise ValueError(
                "InterpretationBoundary requires a non-empty hypothesis"
            )

        if not self.measurement:
            raise ValueError(
                "InterpretationBoundary requires a non-empty measurement"
            )

        for name, value in (
            ("normal_reference", self.normal_reference),
            ("observable_boundary", self.observable_boundary),
            ("substantial_boundary", self.substantial_boundary),
        ):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be numeric"
                )

            if not isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite"
                )

        if self.direction not in {-1, 1}:
            raise ValueError(
                "InterpretationBoundary direction must be either -1 or 1"
            )

        if self.direction == 1:
            if not (
                self.normal_reference
                <= self.observable_boundary
                <= self.substantial_boundary
            ):
                raise ValueError(
                    "Increasing evidence boundaries must be ordered "
                    "normal <= observable <= substantial"
                )

        else:
            if not (
                self.normal_reference
                >= self.observable_boundary
                >= self.substantial_boundary
            ):
                raise ValueError(
                    "Decreasing evidence boundaries must be ordered "
                    "normal >= observable >= substantial"
                )

    def classify(self, value: float) -> str:
        """
        Classify an evidence value using the learned boundaries.

        The returned labels describe evidence strength, not posterior
        probability or causal certainty.
        """
        if not isinstance(value, (int, float)):
            raise ValueError("value must be numeric")

        value = float(value)

        if not isfinite(value):
            raise ValueError("value must be finite")

        if self.direction == 1:
            if value >= self.substantial_boundary:
                return "substantial"

            if value >= self.observable_boundary:
                return "observable"

            return "limited"

        if value <= self.substantial_boundary:
            return "substantial"

        if value <= self.observable_boundary:
            return "observable"

        return "limited"


class EvidenceInterpretationCalibrator:
    """
    Learns evidence-interpretation boundaries from calibration observations.

    Calibration observations must contain both fault and non-fault examples
    for the requested evidence group / hypothesis / measurement combination.

    The calibrator is deliberately separate from Bayesian likelihood
    calibration. Its output is presentation metadata only and must never be
    used as a likelihood or posterior directly.
    """

    def __init__(
        self,
        *,
        normal_quantile: float = 0.95,
    ) -> None:
        if not 0.5 < normal_quantile < 1.0:
            raise ValueError(
                "normal_quantile must be between 0.5 and 1.0"
            )

        self.normal_quantile = float(normal_quantile)
        self._boundaries: dict[
            tuple[str, str, str],
            InterpretationBoundary,
        ] = {}

    def fit(
        self,
        observations: Iterable[CalibrationObservation],
    ) -> None:
        observations = tuple(observations)

        if not observations:
            raise ValueError(
                "observations cannot be empty"
            )

        grouped: dict[
            tuple[str, str, str],
            list[CalibrationObservation],
        ] = {}

        for observation in observations:
            key = (
                observation.evidence_group,
                observation.hypothesis,
                observation.measurement,
            )

            grouped.setdefault(key, []).append(observation)

        boundaries: dict[
            tuple[str, str, str],
            InterpretationBoundary,
        ] = {}

        for key, group in grouped.items():
            true_values = [
                float(observation.value)
                for observation in group
                if observation.hypothesis_true
            ]

            false_values = [
                float(observation.value)
                for observation in group
                if not observation.hypothesis_true
            ]

            if len(true_values) < 2:
                raise ValueError(
                    f"At least two fault calibration observations are required "
                    f"for {key}"
                )

            if len(false_values) < 2:
                raise ValueError(
                    f"At least two non-fault calibration observations are "
                    f"required for {key}"
                )

            boundary = self._build_boundary(
                evidence_group=key[0],
                hypothesis=key[1],
                measurement=key[2],
                true_values=true_values,
                false_values=false_values,
            )

            boundaries[key] = boundary

        self._boundaries = boundaries

    def get(
        self,
        evidence_group: str,
        hypothesis: str,
        measurement: str = "default",
    ) -> InterpretationBoundary:
        key = (
            evidence_group,
            hypothesis,
            measurement,
        )

        try:
            return self._boundaries[key]
        except KeyError as exc:
            raise KeyError(
                "No interpretation boundary is available for "
                f"{key}"
            ) from exc

    def contains(
        self,
        evidence_group: str,
        hypothesis: str,
        measurement: str = "default",
    ) -> bool:
        return (
            evidence_group,
            hypothesis,
            measurement,
        ) in self._boundaries

    def boundaries(
        self,
    ) -> dict[
        tuple[str, str, str],
        InterpretationBoundary,
    ]:
        return dict(self._boundaries)

    def classify(
        self,
        *,
        evidence_group: str,
        hypothesis: str,
        value: float,
        measurement: str = "default",
    ) -> str:
        boundary = self.get(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            measurement=measurement,
        )

        return boundary.classify(value)

    def _build_boundary(
        self,
        *,
        evidence_group: str,
        hypothesis: str,
        measurement: str,
        true_values: list[float],
        false_values: list[float],
    ) -> InterpretationBoundary:
        true_median = median(true_values)
        false_median = median(false_values)

        if true_median == false_median:
            raise ValueError(
                "Calibration distributions do not provide a directional "
                f"separation for {(evidence_group, hypothesis, measurement)}"
            )

        direction = (
            1
            if true_median > false_median
            else -1
        )

        normal_reference = median(false_values)

        observable_boundary = self._quantile(
            false_values,
            self.normal_quantile,
        )

        # The central fault-like value gives us a second empirical boundary.
        substantial_boundary = true_median

        if direction == 1:
            # If the distributions overlap, keep the ordering valid while
            # retaining the empirical normal upper boundary.
            if substantial_boundary < observable_boundary:
                substantial_boundary = observable_boundary

        else:
            # For decreasing fault evidence, the normal lower-tail boundary
            # is obtained by mirroring the upper quantile around the median.
            normal_lower_boundary = self._lower_quantile(
                false_values,
                self.normal_quantile,
            )

            observable_boundary = normal_lower_boundary

            if substantial_boundary > observable_boundary:
                substantial_boundary = observable_boundary

        return InterpretationBoundary(
            evidence_group=evidence_group,
            hypothesis=hypothesis,
            measurement=measurement,
            normal_reference=normal_reference,
            observable_boundary=observable_boundary,
            substantial_boundary=substantial_boundary,
            direction=direction,
        )

    @staticmethod
    def _quantile(
        values: list[float],
        quantile: float,
    ) -> float:
        ordered = sorted(float(value) for value in values)

        if len(ordered) == 1:
            return ordered[0]

        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower

        return (
            ordered[lower]
            + fraction * (ordered[upper] - ordered[lower])
        )

    @classmethod
    def _lower_quantile(
        cls,
        values: list[float],
        quantile: float,
    ) -> float:
        return cls._quantile(
            values,
            1.0 - quantile,
        )