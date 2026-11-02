from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CalibrationObservation:
    evidence_group: str
    hypothesis: str
    hypothesis_true: bool
    value: float
    measurement: str = "default"

    def __post_init__(self):
        if not self.evidence_group:
            raise ValueError(
                "evidence_group must not be empty"
            )

        if not self.hypothesis:
            raise ValueError(
                "hypothesis must not be empty"
            )

        if not isinstance(self.hypothesis_true, bool):
            raise ValueError(
                "hypothesis_true must be a boolean"
            )

        if not isinstance(self.value, (int, float)):
            raise ValueError(
                "value must be numeric"
            )

        if not math.isfinite(float(self.value)):
            raise ValueError(
                "value must be finite"
            )

        if not self.measurement:
            raise ValueError(
                "measurement must not be empty"
            )