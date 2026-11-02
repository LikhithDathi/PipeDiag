from __future__ import annotations

from dataclasses import dataclass

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.diagnostic_engine import DiagnosticEngine
from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.scenario_runner import ScenarioRunner

from experiments.distribution_shift import (
    run_distribution_shift_pipeline,
)
from experiments.distribution_shift_evidence import (
    collect_distribution_shift_evidence,
)
from experiments.full_system_validation import (
    HYPOTHESIS_IDS,
    build_calibrated_likelihood_registry,
)
from experiments.imbalance_evidence import (
    collect_imbalance_evidence,
)
from experiments.imbalance_pipeline import (
    run_imbalance_pipeline,
)


@dataclass(frozen=True)
class MultiFaultValidation:
    result: DiagnosticResult

    @property
    def posterior(self) -> dict[str, float]:
        return self.result.posterior

    @property
    def uncertainty(self) -> float:
        return self.result.uncertainty

    @property
    def predicted_hypothesis(self) -> str:
        return max(
            self.result.posterior,
            key=self.result.posterior.get,
        )

    @property
    def top_two_hypotheses(self) -> list[str]:
        return [
            hypothesis
            for hypothesis, _ in sorted(
                self.result.posterior.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:2]
        ]


class DistributionShiftImbalanceEvidenceProvider:
    def collect(
        self,
        scenario: DiagnosticScenario,
    ) -> list:
        n_samples = scenario.configuration.get(
            "n_samples",
            500,
        )
        n_features = scenario.configuration.get(
            "n_features",
            10,
        )
        test_size = scenario.configuration.get(
            "test_size",
            0.2,
        )
        shift_magnitude = scenario.configuration.get(
            "shift_magnitude",
            1.5,
        )
        minority_fraction = scenario.configuration.get(
            "minority_fraction",
            0.10,
        )
        random_state = scenario.configuration.get(
            "random_state",
            42,
        )

        distribution_shift_result = (
            run_distribution_shift_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                test_size=test_size,
                shift_magnitude=shift_magnitude,
                random_state=random_state,
            )
        )

        imbalance_result = run_imbalance_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            minority_fraction=minority_fraction,
            test_size=test_size,
            random_state=random_state,
        )

        return [
            collect_distribution_shift_evidence(
                distribution_shift_result,
            ),
            collect_imbalance_evidence(
                imbalance_result,
            ),
        ]


def build_multi_fault_engine() -> DiagnosticEngine:
    registry = build_calibrated_likelihood_registry()

    bayesian_inference = BayesianInference(
        hypothesis_ids=list(HYPOTHESIS_IDS),
    )

    likelihood_evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    bayesian_evidence_inference = BayesianEvidenceInference(
        bayesian_inference=bayesian_inference,
    )

    return DiagnosticEngine(
        likelihood_evaluator=likelihood_evaluator,
        bayesian_evidence_inference=bayesian_evidence_inference,
    )


def build_distribution_shift_imbalance_scenario(
    *,
    random_state: int = 42,
) -> DiagnosticScenario:
    return DiagnosticScenario(
        scenario_id="multi_fault_H3_H4",
        expected_hypothesis=None,
        description=(
            "Controlled multi-fault scenario combining "
            "moderate distribution shift and class imbalance."
        ),
        configuration={
            "n_samples": 500,
            "n_features": 10,
            "test_size": 0.2,
            "shift_magnitude": 0.5,
            "minority_fraction": 0.10,
            "random_state": random_state,
        },
        metadata={
            "scenario_type": "multi_fault",
            "faults_injected": ("H3", "H4"),
        },
    )


def run_distribution_shift_imbalance_validation(
    *,
    random_state: int = 42,
) -> MultiFaultValidation:
    scenario = build_distribution_shift_imbalance_scenario(
        random_state=random_state,
    )

    evidence_provider = (
        DistributionShiftImbalanceEvidenceProvider()
    )

    diagnostic_engine = build_multi_fault_engine()

    runner = ScenarioRunner(
        evidence_provider=evidence_provider,
        diagnostic_engine=diagnostic_engine,
    )

    result = runner.run(scenario)

    return MultiFaultValidation(
        result=result,
    )