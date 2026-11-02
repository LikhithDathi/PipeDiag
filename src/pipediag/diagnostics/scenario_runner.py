from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.scenario_result import (
    ScenarioResult,
)


class ScenarioRunner:
    def __init__(
        self,
        evidence_provider,
        diagnostic_engine,
    ):
        if evidence_provider is None:
            raise ValueError(
                "ScenarioRunner requires an evidence provider"
            )

        if diagnostic_engine is None:
            raise ValueError(
                "ScenarioRunner requires a diagnostic engine"
            )

        self.evidence_provider = evidence_provider
        self.diagnostic_engine = diagnostic_engine

    def run(
        self,
        scenario: DiagnosticScenario,
    ) -> ScenarioResult:
        if scenario is None:
            raise ValueError(
                "ScenarioRunner requires a scenario"
            )

        evidence = self.evidence_provider.collect(
            scenario
        )

        diagnostic_result = self.diagnostic_engine.diagnose(
            evidence_groups=evidence,
            metadata=scenario.metadata,
        )

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            expected_hypothesis=scenario.expected_hypothesis,
            posterior=diagnostic_result.posterior,
            uncertainty=diagnostic_result.uncertainty,
            metadata=scenario.metadata,
        )