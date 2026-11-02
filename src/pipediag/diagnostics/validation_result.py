from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    scenario_id: str
    expected_hypothesis: str | None
    predicted_hypothesis: str
    expected_posterior: float | None
    uncertainty: float
    posterior: dict[str, float]

    @property
    def is_correct(self) -> bool | None:
        if self.expected_hypothesis is None:
            return None

        return (
            self.predicted_hypothesis
            == self.expected_hypothesis
        )

    def __post_init__(self):
        if not self.scenario_id:
            raise ValueError(
                "ValidationResult requires a non-empty scenario_id"
            )

        if not self.predicted_hypothesis:
            raise ValueError(
                "ValidationResult requires a predicted_hypothesis"
            )

        if not self.posterior:
            raise ValueError(
                "ValidationResult requires a non-empty posterior"
            )

        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(
                "uncertainty must be between 0 and 1"
            )

        for hypothesis, probability in self.posterior.items():
            if not hypothesis:
                raise ValueError(
                    "Posterior hypothesis names must be non-empty"
                )

            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "Posterior probabilities must be between 0 and 1"
                )

        if self.expected_hypothesis is not None:
            if self.expected_hypothesis not in self.posterior:
                raise ValueError(
                    "expected_hypothesis must be present in posterior"
                )

            if self.expected_posterior is None:
                raise ValueError(
                    "expected_posterior is required when "
                    "expected_hypothesis is provided"
                )

            if not 0.0 <= self.expected_posterior <= 1.0:
                raise ValueError(
                    "expected_posterior must be between 0 and 1"
                )

        elif self.expected_posterior is not None:
            raise ValueError(
                "expected_posterior must be None when "
                "expected_hypothesis is None"
            )