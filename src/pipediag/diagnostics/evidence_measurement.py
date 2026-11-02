from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EvidenceMeasurement:
    name: str
    value: float

    def __post_init__(self):
        if not self.name:
            raise ValueError(
                "EvidenceMeasurement requires a non-empty name"
            )

        if not isinstance(self.value, (int, float)):
            raise ValueError(
                "EvidenceMeasurement value must be numeric"
            )

        if not math.isfinite(float(self.value)):
            raise ValueError(
                "EvidenceMeasurement value must be finite"
            )