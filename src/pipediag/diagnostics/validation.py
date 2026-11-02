from __future__ import annotations

from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.scenario_runner import (
    ScenarioRunner,
)
from pipediag.diagnostics.validation_result import (
    ValidationResult,
)


class DiagnosticValidator:
    def __init__(self, scenario_runner: ScenarioRunner):
        if scenario_runner is None:
            raise ValueError(
                "DiagnosticValidator requires a scenario runner"
            )

        self.scenario_runner = scenario_runner

    def validate(
        self,
        scenarios: list[DiagnosticScenario],
    ) -> list[ValidationResult]:
        if scenarios is None:
            raise ValueError("scenarios are required")

        results: list[ValidationResult] = []
        seen_ids: set[str] = set()

        for scenario in scenarios:
            if scenario is None:
                raise ValueError(
                    "scenarios must not contain None"
                )

            if scenario.scenario_id in seen_ids:
                raise ValueError(
                    f"Duplicate scenario "
                    f"'{scenario.scenario_id}'"
                )

            seen_ids.add(scenario.scenario_id)

            result = self.scenario_runner.run(scenario)

            predicted_hypothesis = result.predicted_hypothesis

            expected_posterior = None

            if result.expected_hypothesis is not None:
                expected_posterior = result.posterior[
                    result.expected_hypothesis
                ]

            results.append(
                ValidationResult(
                    scenario_id=result.scenario_id,
                    expected_hypothesis=result.expected_hypothesis,
                    predicted_hypothesis=predicted_hypothesis,
                    expected_posterior=expected_posterior,
                    uncertainty=result.uncertainty,
                    posterior=dict(result.posterior),
                )
            )

        return results