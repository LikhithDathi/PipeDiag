from __future__ import annotations

from dataclasses import dataclass

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.diagnostic_engine import DiagnosticEngine
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.scenario_runner import ScenarioRunner

from experiments.data_quality_evidence import (
    collect_data_quality_evidence,
)
from experiments.data_quality_pipeline import (
    run_data_quality_pipeline,
)
from experiments.feature_problem_evidence import (
    collect_feature_problem_evidence,
)
from experiments.feature_problem_pipeline import (
    run_feature_problem_pipeline,
)
from experiments.full_system_validation import (
    HYPOTHESIS_IDS,
    build_calibrated_likelihood_registry,
)
from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.imbalance_evidence import (
    collect_imbalance_evidence,
)
from experiments.imbalance_pipeline import (
    run_imbalance_pipeline,
)
from experiments.overfitting_pipeline import (
    run_overfitting_pipeline,
)


@dataclass(frozen=True)
class CleanBaselineValidation:
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
    def maximum_posterior(self) -> float:
        return max(self.result.posterior.values())

    @property
    def has_dominant_hypothesis(self) -> bool:
        """
        A clean pipeline should not produce a dominant
        diagnostic hypothesis.

        This criterion is deliberately separate from the
        eventual user-facing reporting threshold.
        """
        return self.maximum_posterior > 0.5

    @property
    def is_uncertain(self) -> bool:
        """
        A clean negative control should retain substantial
        diagnostic uncertainty because no fault was injected.
        """
        return self.uncertainty > 0.5

    @property
    def passed(self) -> bool:
        """
        Clean-control validation passes when the system does
        not strongly favor a single fault hypothesis and
        retains substantial uncertainty.
        """
        return (
            not self.has_dominant_hypothesis
            and self.is_uncertain
        )


class CleanBaselineEvidenceProvider:
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
        random_state = scenario.configuration.get(
            "random_state",
            42,
        )

        generalization_result = run_overfitting_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            test_size=test_size,
            random_state=random_state,
            max_depth=2,
        )

        imbalance_result = run_imbalance_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            minority_fraction=0.40,
            test_size=test_size,
            random_state=random_state,
        )

        data_quality_result = run_data_quality_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            test_size=test_size,
            missing_rate=0.0,
            random_state=random_state,
        )

        feature_result = run_feature_problem_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            constant_feature_ratio=0.0,
            random_state=random_state,
        )

        return [
            collect_generalization_evidence(
                generalization_result,
            ),
            collect_imbalance_evidence(
                imbalance_result,
            ),
            collect_data_quality_evidence(
                data_quality_result,
            ),
            collect_feature_problem_evidence(
                feature_result,
            ),
        ]


def build_clean_baseline_engine() -> DiagnosticEngine:
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


def build_clean_baseline_scenario(
    *,
    random_state: int = 42,
) -> DiagnosticScenario:
    return DiagnosticScenario(
        scenario_id="clean_baseline",
        expected_hypothesis=None,
        description=(
            "Clean ML pipeline with no intentionally "
            "injected diagnostic fault."
        ),
        configuration={
            "n_samples": 500,
            "n_features": 10,
            "test_size": 0.2,
            "random_state": random_state,
        },
        metadata={
            "scenario_type": "negative_control",
            "fault_injected": False,
        },
    )


def run_clean_baseline_validation(
    *,
    random_state: int = 42,
) -> CleanBaselineValidation:
    scenario = build_clean_baseline_scenario(
        random_state=random_state,
    )

    evidence_provider = CleanBaselineEvidenceProvider()

    diagnostic_engine = build_clean_baseline_engine()

    runner = ScenarioRunner(
        evidence_provider=evidence_provider,
        diagnostic_engine=diagnostic_engine,
    )

    result = runner.run(scenario)

    return CleanBaselineValidation(
        result=result,
    )