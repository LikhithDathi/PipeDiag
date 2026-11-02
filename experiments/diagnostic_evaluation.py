from __future__ import annotations

from dataclasses import dataclass

from experiments.full_system_validation import (
    SingleFaultEvidenceProvider,
    build_diagnostic_engine,
    build_single_fault_scenarios,
)
from pipediag.diagnostics.evaluation import (
    DiagnosticEvaluation,
    evaluate_diagnostic_results,
)
from pipediag.diagnostics.scenario_runner import ScenarioRunner


@dataclass(frozen=True)
class SingleFaultDiagnosticEvaluation:
    evaluation: DiagnosticEvaluation
    scenario_results: tuple


def run_single_fault_diagnostic_evaluation(
    *,
    random_state: int = 42,
) -> SingleFaultDiagnosticEvaluation:
    scenarios = build_single_fault_scenarios(
        random_state=random_state,
    )

    engine = build_diagnostic_engine()

    provider = SingleFaultEvidenceProvider()

    runner = ScenarioRunner(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    results = tuple(
        runner.run(scenario)
        for scenario in scenarios
    )

    evaluation = evaluate_diagnostic_results(
        list(results),
    )

    return SingleFaultDiagnosticEvaluation(
        evaluation=evaluation,
        scenario_results=results,
    )