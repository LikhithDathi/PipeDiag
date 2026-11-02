from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiagnosticScenario:
    scenario_id: str
    expected_hypothesis: str | None
    description: str
    configuration: dict[str, Any] = field(
        default_factory=dict
    )
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.scenario_id:
            raise ValueError(
                "DiagnosticScenario requires a non-empty scenario_id"
            )

        if not self.description:
            raise ValueError(
                "DiagnosticScenario requires a non-empty description"
            )

        if self.expected_hypothesis == "":
            raise ValueError(
                "expected_hypothesis must be non-empty when provided"
            )

        object.__setattr__(
            self,
            "configuration",
            dict(self.configuration),
        )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )