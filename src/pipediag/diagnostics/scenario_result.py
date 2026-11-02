from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    expected_hypothesis: str | None
    posterior: dict[str, float]
    uncertainty: float
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.scenario_id:
            raise ValueError(
                "ScenarioResult requires a non-empty scenario_id"
            )

        if not self.posterior:
            raise ValueError(
                "ScenarioResult requires a non-empty posterior"
            )

        for probability in self.posterior.values():
            if not math.isfinite(probability):
                raise ValueError(
                    "Posterior probabilities must be finite"
                )

            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "Posterior probabilities must be between 0 and 1"
                )

        if not math.isclose(
            sum(self.posterior.values()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Posterior probabilities must sum to one"
            )

        if not math.isfinite(self.uncertainty):
            raise ValueError(
                "Uncertainty must be finite"
            )

        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(
                "Uncertainty must be between 0 and 1"
            )

        if self.expected_hypothesis == "":
            raise ValueError(
                "expected_hypothesis must be non-empty when provided"
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def predicted_hypothesis(self) -> str:
        return max(
            self.posterior,
            key=self.posterior.get,
        )

    @property
    def is_correct(self) -> bool | None:
        if self.expected_hypothesis is None:
            return None

        return (
            self.predicted_hypothesis
            == self.expected_hypothesis
        )