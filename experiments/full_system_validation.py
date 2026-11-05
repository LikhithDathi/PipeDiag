from __future__ import annotations

from dataclasses import dataclass

from experiments.data_quality_calibration import (
    collect_data_quality_calibration_observations,
)
from experiments.data_quality_evidence import (
    collect_data_quality_evidence,
)
from experiments.data_quality_pipeline import (
    run_data_quality_pipeline,
)

from experiments.distribution_shift_calibration import (
    collect_distribution_shift_calibration_observations,
)
from experiments.runtime_leakage_calibration import (
    collect_runtime_leakage_calibration_observations,
)
from experiments.distribution_shift_evidence import (
    collect_distribution_shift_evidence,
)
from experiments.distribution_shift import (
    run_distribution_shift_pipeline,
)
from experiments.evaluation_problem_calibration import (
    collect_evaluation_problem_calibration_observations,
)
from experiments.evaluation_problem_evidence import (
    collect_evaluation_problem_evidence,
)
from experiments.evaluation_problem_pipeline import (
    run_evaluation_problem_pipeline,
)
from experiments.execution_problem_calibration import (
    collect_execution_problem_calibration_observations,
)
from experiments.execution_problem_evidence import (
    collect_execution_problem_evidence,
)
from experiments.execution_problem_pipeline import (
    run_execution_problem_pipeline,
)
from experiments.feature_problem_calibration import (
    collect_feature_problem_calibration_observations,
)
from experiments.feature_problem_evidence import (
    collect_feature_problem_evidence,
)
from experiments.feature_problem_pipeline import (
    run_feature_problem_pipeline,
)
from experiments.generalization_calibration import (
    collect_generalization_calibration_observations,
)
from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.imbalance_calibration import (
    collect_imbalance_calibration_observations,
)
from experiments.imbalance_evidence import (
    collect_imbalance_evidence,
)
from experiments.imbalance_pipeline import (
    run_imbalance_pipeline,
)
from experiments.leakage_calibration import (
    collect_leakage_calibration_observations,
)
from experiments.leakage_evidence import (
    collect_leakage_evidence,
)
from experiments.leakage_pipeline import (
    run_leakage_pipeline,
)
from experiments.overfitting_pipeline import (
    run_overfitting_pipeline,
)
from experiments.underfitting_calibration import (
    collect_underfitting_calibration_observations,
)

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.diagnostic_engine import (
    DiagnosticEngine,
)
from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.hypotheses import (
    DIAGNOSTIC_HYPOTHESES,
)
from pipediag.diagnostics.likelihood_builder import (
    LikelihoodModelBuilder,
)
from pipediag.diagnostics.scenario_runner import (
    ScenarioRunner,
)
from pipediag.diagnostics.validation import (
    DiagnosticValidator,
)
from pipediag.diagnostics.validation_result import (
    ValidationResult,
)


HYPOTHESIS_IDS = tuple(
    hypothesis.id
    for hypothesis in DIAGNOSTIC_HYPOTHESES
)


@dataclass(frozen=True)
class FullSystemValidation:
    results: list[ValidationResult]

    @property
    def total_scenarios(self) -> int:
        return len(self.results)

    @property
    def correct_predictions(self) -> int:
        return sum(
            result.is_correct is True
            for result in self.results
        )

    @property
    def top1_accuracy(self) -> float:
        if not self.results:
            return 0.0

        return (
            self.correct_predictions
            / self.total_scenarios
        )


class SingleFaultEvidenceProvider:
    """
    Adapter from DiagnosticScenario configuration to the
    existing controlled experiment/evidence modules.
    """

    def collect(
        self,
        scenario: DiagnosticScenario,
    ):
        hypothesis = scenario.expected_hypothesis
        configuration = scenario.configuration

        random_state = configuration.get(
            "random_state",
            42,
        )

        n_samples = configuration.get(
            "n_samples",
            500,
        )

        n_features = configuration.get(
            "n_features",
            10,
        )

        if hypothesis == "H1":
            result = run_leakage_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                leakage_strength=configuration.get(
                    "leakage_strength",
                    1.0,
                ),
            )

            return [
                collect_leakage_evidence(result)
            ]

        if hypothesis == "H2":
            result = run_overfitting_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                max_depth=configuration.get(
                    "max_depth",
                    None,
                ),
            )

            return [
                collect_generalization_evidence(result)
            ]

        if hypothesis == "H3":
            result = run_distribution_shift_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                shift_magnitude=configuration.get(
                    "shift_magnitude",
                    1.5,
                ),
            )

            return [
                collect_distribution_shift_evidence(
                    result
                )
            ]

        if hypothesis == "H4":
            result = run_imbalance_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                minority_fraction=configuration.get(
                    "minority_fraction",
                    0.10,
                ),
            )

            return [
                collect_imbalance_evidence(result)
            ]

        if hypothesis == "H5":
            result = run_data_quality_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                missing_rate=configuration.get(
                    "missing_rate",
                    0.20,
                ),
            )

            return [
                collect_data_quality_evidence(result)
            ]

        if hypothesis == "H6":
            result = run_evaluation_problem_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                test_size=configuration.get(
                    "test_size",
                    0.03,
                ),
            )

            return [
                collect_evaluation_problem_evidence(
                    result
                )
            ]

        if hypothesis == "H7":
            result = run_overfitting_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                max_depth=configuration.get(
                    "max_depth",
                    1,
                ),
            )

            return [
                collect_generalization_evidence(result)
            ]

        if hypothesis == "H8":
            result = run_feature_problem_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                constant_feature_ratio=configuration.get(
                    "constant_feature_ratio",
                    0.70,
                ),
            )

            return [
                collect_feature_problem_evidence(result)
            ]

        if hypothesis == "H9":
            result = run_execution_problem_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
                inject_failure=True,
                failure_event_count=configuration.get(
                    "failure_event_count",
                    1,
                ),
                total_execution_events=configuration.get(
                    "total_execution_events",
                    3,
                ),
            )

            return [
                collect_execution_problem_evidence(
                    result
                )
            ]

        raise ValueError(
            f"Unsupported hypothesis '{hypothesis}'"
        )


def build_calibrated_likelihood_registry():
    """
    Build one likelihood registry from all controlled
    calibration observations.
    """

    observations = []

    observations.extend(
        collect_leakage_calibration_observations()
    )

    observations.extend(
        collect_runtime_leakage_calibration_observations()
    )

    observations.extend(
        collect_generalization_calibration_observations()
    )

    observations.extend(
        collect_distribution_shift_calibration_observations()
    )

    observations.extend(
        collect_imbalance_calibration_observations()
    )

    observations.extend(
        collect_data_quality_calibration_observations()
    )

    observations.extend(
        collect_evaluation_problem_calibration_observations()
    )

    observations.extend(
        collect_underfitting_calibration_observations()
    )

    observations.extend(
        collect_feature_problem_calibration_observations()
    )

    observations.extend(
        collect_execution_problem_calibration_observations()
    )

    return LikelihoodModelBuilder().build(
        observations
    )


def build_diagnostic_engine():
    registry = (
        build_calibrated_likelihood_registry()
    )

    bayesian_inference = BayesianInference(
        hypothesis_ids=list(HYPOTHESIS_IDS)
    )

    likelihood_evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=(
            EVIDENCE_HYPOTHESIS_MAP
        ),
    )

    bayesian_evidence_inference = (
        BayesianEvidenceInference(
            bayesian_inference=bayesian_inference
        )
    )

    return DiagnosticEngine(
        likelihood_evaluator=likelihood_evaluator,
        bayesian_evidence_inference=(
            bayesian_evidence_inference
        ),
    )


def build_single_fault_scenarios(
    random_state: int = 42,
) -> list[DiagnosticScenario]:
    return [
        DiagnosticScenario(
            scenario_id="single_fault_H1",
            expected_hypothesis="H1",
            description="Controlled data leakage scenario.",
            configuration={
                "random_state": random_state,
                "leakage_strength": 1.0,
            },
        ),
        DiagnosticScenario(
            scenario_id="single_fault_H2",
            expected_hypothesis="H2",
            description="Controlled generalization failure scenario.",
            configuration={
                "random_state": random_state,
                "max_depth": None,
            },
        ),
        DiagnosticScenario(
            scenario_id="single_fault_H3",
            expected_hypothesis="H3",
            description="Controlled distribution shift scenario.",
            configuration={
                "random_state": random_state,
                "shift_magnitude": 1.5,
            },
        ),
        DiagnosticScenario(
            scenario_id="single_fault_H4",
            expected_hypothesis="H4",
            description="Controlled class imbalance scenario.",
            configuration={
                "random_state": random_state,
                "minority_fraction": 0.10,
            },
        ),
        DiagnosticScenario(
            scenario_id="single_fault_H5",
            expected_hypothesis="H5",
            description="Controlled data quality scenario.",
            configuration={
                "random_state": random_state,
                "missing_rate": 0.20,
            },
        ),
        DiagnosticScenario(
            scenario_id="single_fault_H6",
            expected_hypothesis="H6",
            description="Controlled evaluation reliability scenario.",
            configuration={
                "random_state": random_state,
                "test_size": 0.03,
            },
        ),
        DiagnosticScenario(
            scenario_id="single_fault_H7",
            expected_hypothesis="H7",
            description="Controlled underfitting scenario.",
            configuration={
                "random_state": random_state,
                "max_depth": 1,
            },
        ),
        DiagnosticScenario(
            scenario_id="single_fault_H8",
            expected_hypothesis="H8",
            description="Controlled feature representation scenario.",
            configuration={
                "random_state": random_state,
                "constant_feature_ratio": 0.70,
            },
        ),
        DiagnosticScenario(
            scenario_id="single_fault_H9",
            expected_hypothesis="H9",
            description="Controlled execution anomaly scenario.",
            configuration={
                "random_state": random_state,
                "failure_event_count": 1,
                "total_execution_events": 3,
            },
        ),
    ]


def run_single_fault_validation(
    random_state: int = 42,
) -> FullSystemValidation:
    diagnostic_engine = build_diagnostic_engine()

    evidence_provider = SingleFaultEvidenceProvider()

    scenario_runner = ScenarioRunner(
        evidence_provider=evidence_provider,
        diagnostic_engine=diagnostic_engine,
    )

    validator = DiagnosticValidator(
        scenario_runner
    )

    scenarios = build_single_fault_scenarios(
        random_state=random_state
    )

    results = validator.validate(
        scenarios
    )

    return FullSystemValidation(
        results=results
    )