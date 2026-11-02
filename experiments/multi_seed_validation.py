from __future__ import annotations

from dataclasses import dataclass

from pipediag.diagnostics.diagnostic_scenario import DiagnosticScenario
from pipediag.diagnostics.scenario_result import ScenarioResult
from pipediag.diagnostics.scenario_runner import ScenarioRunner

from experiments.full_system_validation import (
    HYPOTHESIS_IDS,
    SingleFaultEvidenceProvider,
    build_diagnostic_engine,
)


VALIDATION_SEEDS = tuple(range(100, 110))


@dataclass(frozen=True)
class MultiSeedValidation:
    results: tuple[ScenarioResult, ...]
    total_scenarios: int
    correct_predictions: int
    top1_accuracy: float

    @property
    def incorrect_predictions(self) -> int:
        return self.total_scenarios - self.correct_predictions


def build_multi_seed_scenarios() -> tuple[DiagnosticScenario, ...]:
    scenarios: list[DiagnosticScenario] = []

    for random_state in VALIDATION_SEEDS:
        for hypothesis in HYPOTHESIS_IDS:
            if hypothesis == "H1":
                configuration = {
                    "leakage_strength": 1.0,
                    "random_state": random_state,
                }

            elif hypothesis == "H2":
                configuration = {
                    "max_depth": None,
                    "random_state": random_state,
                }

            elif hypothesis == "H3":
                configuration = {
                    "shift_magnitude": 1.5,
                    "random_state": random_state,
                }

            elif hypothesis == "H4":
                configuration = {
                    "minority_fraction": 0.10,
                    "random_state": random_state,
                }

            elif hypothesis == "H5":
                configuration = {
                    "missing_rate": 0.20,
                    "random_state": random_state,
                }

            elif hypothesis == "H6":
                configuration = {
                    "test_size": 0.03,
                    "random_state": random_state,
                }

            elif hypothesis == "H7":
                configuration = {
                    "max_depth": 1,
                    "random_state": random_state,
                }

            elif hypothesis == "H8":
                configuration = {
                    "constant_feature_ratio": 0.70,
                    "random_state": random_state,
                }

            elif hypothesis == "H9":
                configuration = {
                    "failure_event_count": 1,
                    "total_execution_events": 3,
                    "random_state": random_state,
                }

            else:
                raise ValueError(f"Unsupported hypothesis: {hypothesis}")

            scenarios.append(
                DiagnosticScenario(
                    scenario_id=f"multi_seed_{hypothesis}_{random_state}",
                    expected_hypothesis=hypothesis,
                    description=(
                        f"Independent validation scenario for {hypothesis} "
                        f"using random state {random_state}."
                    ),
                    configuration=configuration,
                    metadata={
                        "scenario_type": "independent_single_fault",
                        "validation_seed": random_state,
                        "fault_injected": True,
                    },
                )
            )

    return tuple(scenarios)


def run_multi_seed_validation() -> MultiSeedValidation:
    scenarios = build_multi_seed_scenarios()

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

    correct_predictions = sum(
        result.is_correct
        for result in results
        if result.is_correct is not None
    )

    evaluated_results = [
        result
        for result in results
        if result.expected_hypothesis is not None
    ]

    total_scenarios = len(evaluated_results)

    top1_accuracy = (
        correct_predictions / total_scenarios
        if total_scenarios
        else 0.0
    )

    return MultiSeedValidation(
        results=results,
        total_scenarios=total_scenarios,
        correct_predictions=correct_predictions,
        top1_accuracy=top1_accuracy,
    )