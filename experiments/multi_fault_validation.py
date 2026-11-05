
from __future__ import annotations

from dataclasses import dataclass

from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.multi_fault_bayesian import (
    MultiFaultBayesianInference,
)
from pipediag.diagnostics.multi_fault_diagnostic_engine import (
    MultiFaultDiagnosticEngine,
)
from pipediag.diagnostics.multi_fault_diagnostic_result import (
    MultiFaultDiagnosticResult,
)

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
    """Result wrapper for a multi-fault validation experiment."""

    result: MultiFaultDiagnosticResult

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
            for hypothesis, _ in self.result.ranked_hypotheses[:2]
        ]

    @property
    def detected_hypotheses(self) -> frozenset[str]:
        return self.result.detected_hypotheses


class DistributionShiftImbalanceEvidenceProvider:
    """Collects G3 and G4 evidence for the multi-fault scenario."""

    def collect(
        self,
        scenario: DiagnosticScenario,
    ) -> list:

        configuration = scenario.configuration

        n_samples = configuration.get("n_samples", 500)
        n_features = configuration.get("n_features", 10)
        test_size = configuration.get("test_size", 0.2)
        shift_magnitude = configuration.get(
            "shift_magnitude",
            0.5,
        )
        minority_fraction = configuration.get(
            "minority_fraction",
            0.10,
        )
        random_state = configuration.get(
            "random_state",
            42,
        )

        distribution_shift_result = run_distribution_shift_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            test_size=test_size,
            shift_magnitude=shift_magnitude,
            random_state=random_state,
        )

        imbalance_result = run_imbalance_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            minority_fraction=minority_fraction,
            test_size=test_size,
            random_state=random_state,
        )

        distribution_evidence = collect_distribution_shift_evidence(
            distribution_shift_result,
        )

        imbalance_evidence = collect_imbalance_evidence(
            imbalance_result,
        )

        return [
            distribution_evidence,
            imbalance_evidence,
        ]


def build_multi_fault_engine(
    priors: dict[str, float] | None = None,
    detection_threshold: float = 0.5,
) -> MultiFaultDiagnosticEngine:
    """Builds the independent multi-fault diagnostic engine."""

    registry = build_calibrated_likelihood_registry()

    likelihood_evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    bayesian_inference = MultiFaultBayesianInference(
        hypothesis_ids=HYPOTHESIS_IDS,
        priors=priors,
    )

    return MultiFaultDiagnosticEngine(
        likelihood_evaluator=likelihood_evaluator,
        bayesian_inference=bayesian_inference,
        detection_threshold=detection_threshold,
    )


def build_distribution_shift_imbalance_scenario(
    *,
    random_state: int = 42,
) -> DiagnosticScenario:
    """Creates a scenario with both H3 and H4 injected."""

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
    priors: dict[str, float] | None = None,
    detection_threshold: float = 0.5,
) -> MultiFaultValidation:
    """Runs the complete H3 + H4 multi-fault experiment."""

    scenario = build_distribution_shift_imbalance_scenario(
        random_state=random_state,
    )

    evidence_provider = DistributionShiftImbalanceEvidenceProvider()

    evidence = evidence_provider.collect(scenario)

    diagnostic_engine = build_multi_fault_engine(
        priors=priors,
        detection_threshold=detection_threshold,
    )

    result = diagnostic_engine.diagnose(
        evidence_groups=evidence,
        metadata={
            **scenario.metadata,
            "scenario_id": scenario.scenario_id,
        },
    )

    return MultiFaultValidation(result=result)
