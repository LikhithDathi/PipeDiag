from __future__ import annotations

from dataclasses import dataclass

from experiments.data_quality_evidence import collect_data_quality_evidence
from experiments.data_quality_pipeline import run_data_quality_pipeline
from experiments.distribution_shift import run_distribution_shift_pipeline
from experiments.distribution_shift_evidence import (
    collect_distribution_shift_evidence,
)
from experiments.execution_problem_evidence import (
    collect_execution_problem_evidence,
)
from experiments.execution_problem_pipeline import (
    run_execution_problem_pipeline,
)
from experiments.feature_problem_evidence import (
    collect_feature_problem_evidence,
)
from experiments.feature_problem_pipeline import (
    run_feature_problem_pipeline,
)
from experiments.full_system_validation import build_diagnostic_engine
from experiments.imbalance_evidence import collect_imbalance_evidence
from experiments.imbalance_pipeline import run_imbalance_pipeline
from experiments.leakage_evidence import collect_leakage_evidence
from experiments.leakage_pipeline import run_leakage_pipeline


@dataclass(frozen=True)
class FaultStrengthObservation:
    hypothesis: str
    strength: float
    posterior: float
    uncertainty: float
    evidence_value: float | None = None


@dataclass(frozen=True)
class FaultStrengthValidation:
    observations: tuple[FaultStrengthObservation, ...]

    def for_hypothesis(
        self,
        hypothesis: str,
    ) -> tuple[FaultStrengthObservation, ...]:
        return tuple(
            observation
            for observation in self.observations
            if observation.hypothesis == hypothesis
        )

    def evidence_values(
        self,
        hypothesis: str,
    ) -> tuple[float, ...]:
        observations = self.for_hypothesis(hypothesis)

        return tuple(
            float(observation.evidence_value)
            for observation in observations
            if observation.evidence_value is not None
        )

    def evidence_is_monotonic(
        self,
        hypothesis: str,
    ) -> bool:
        observations = self.for_hypothesis(hypothesis)

        if len(observations) < 2:
            return True

        direction = EVIDENCE_DIRECTIONS[hypothesis]

        values = [
            observation.evidence_value
            for observation in observations
        ]

        if any(value is None for value in values):
            return False

        numeric_values = [
            float(value)
            for value in values
        ]

        if direction == 1:
            return all(
                earlier <= later
                for earlier, later in zip(
                    numeric_values,
                    numeric_values[1:],
                )
            )

        return all(
            earlier >= later
            for earlier, later in zip(
                numeric_values,
                numeric_values[1:],
            )
        )

    @property
    def hypotheses(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                observation.hypothesis
                for observation in self.observations
            )
        )

    @property
    def passed(self) -> bool:
        return all(
            self.evidence_is_monotonic(hypothesis)
            for hypothesis in EVIDENCE_DIRECTIONS
        )


# +1 means evidence should increase as fault strength increases.
# -1 means evidence should decrease as fault strength increases.
EVIDENCE_DIRECTIONS = {
    "H1": 1,
    "H3": 1,
    "H4": -1,
    "H5": 1,
    "H8": 1,
    "H9": 1,
}


def _diagnose(
    engine,
    hypothesis: str,
    strength: float,
    evidence,
) -> FaultStrengthObservation:
    result = engine.diagnose(
        evidence_groups=[evidence],
    )

    return FaultStrengthObservation(
        hypothesis=hypothesis,
        strength=strength,
        posterior=result.posterior[hypothesis],
        uncertainty=result.uncertainty,
        evidence_value=float(evidence.value),
    )


def _run_h1(
    engine,
    strengths,
) -> list[FaultStrengthObservation]:
    observations = []

    for strength in strengths:
        # The production leakage pipeline requires a strictly
        # positive leakage strength. A zero-strength validation
        # point is therefore represented by a negligible 0.01
        # injected signal.
        pipeline_strength = max(strength, 0.01)

        result = run_leakage_pipeline(
            n_samples=500,
            n_features=10,
            leakage_strength=pipeline_strength,
            random_state=42,
        )

        evidence = collect_leakage_evidence(result)

        observations.append(
            _diagnose(
                engine,
                "H1",
                strength,
                evidence,
            )
        )

    return observations


def _run_h3(
    engine,
    strengths,
) -> list[FaultStrengthObservation]:
    observations = []

    for strength in strengths:
        result = run_distribution_shift_pipeline(
            n_samples=500,
            n_features=10,
            shift_magnitude=strength,
            random_state=42,
        )

        evidence = collect_distribution_shift_evidence(result)

        observations.append(
            _diagnose(
                engine,
                "H3",
                strength,
                evidence,
            )
        )

    return observations


def _run_h4(
    engine,
    strengths,
) -> list[FaultStrengthObservation]:
    observations = []

    for strength in strengths:
        # Larger validation strength corresponds to a smaller
        # minority-class fraction and therefore stronger imbalance.
        minority_fraction = 0.45 - (0.35 * strength)

        result = run_imbalance_pipeline(
            n_samples=500,
            n_features=10,
            minority_fraction=minority_fraction,
            random_state=42,
        )

        evidence = collect_imbalance_evidence(result)

        observations.append(
            _diagnose(
                engine,
                "H4",
                strength,
                evidence,
            )
        )

    return observations


def _run_h5(
    engine,
    strengths,
) -> list[FaultStrengthObservation]:
    observations = []

    for strength in strengths:
        pipeline_missing_rate = min(strength, 0.99)

        result = run_data_quality_pipeline(
            n_samples=500,
            n_features=10,
            missing_rate=pipeline_missing_rate,
            random_state=42,
        )

        evidence = collect_data_quality_evidence(result)

        observations.append(
            _diagnose(
                engine,
                "H5",
                strength,
                evidence,
            )
        )

    return observations


def _run_h8(
    engine,
    strengths,
) -> list[FaultStrengthObservation]:
    observations = []

    for strength in strengths:
        pipeline_feature_ratio = min(strength, 0.99)

        result = run_feature_problem_pipeline(
            n_samples=500,
            n_features=10,
            constant_feature_ratio=pipeline_feature_ratio,
            random_state=42,
        )

        evidence = collect_feature_problem_evidence(result)

        observations.append(
            _diagnose(
                engine,
                "H8",
                strength,
                evidence,
            )
        )

    return observations


def _run_h9(
    engine,
    strengths,
) -> list[FaultStrengthObservation]:
    observations = []

    total_events = 10

    for strength in strengths:
        failure_count = max(
            1,
            int(round(strength * total_events)),
        )

        failure_count = min(
            failure_count,
            total_events - 1,
        )

        result = run_execution_problem_pipeline(
            n_samples=500,
            n_features=10,
            failure_event_count=failure_count,
            total_execution_events=total_events,
            inject_failure=True,
            random_state=42,
        )

        evidence = collect_execution_problem_evidence(result)

        observations.append(
            _diagnose(
                engine,
                "H9",
                strength,
                evidence,
            )
        )

    return observations


def run_fault_strength_validation(
    *,
    strengths=(0.0, 0.25, 0.50, 0.75, 1.0),
) -> FaultStrengthValidation:
    if not strengths:
        raise ValueError(
            "strengths cannot be empty"
        )

    if any(
        strength < 0.0 or strength > 1.0
        for strength in strengths
    ):
        raise ValueError(
            "all strengths must be between 0 and 1"
        )

    engine = build_diagnostic_engine()

    observations = []

    observations.extend(
        _run_h1(engine, strengths)
    )

    observations.extend(
        _run_h3(engine, strengths)
    )

    observations.extend(
        _run_h4(engine, strengths)
    )

    observations.extend(
        _run_h5(engine, strengths)
    )

    observations.extend(
        _run_h8(engine, strengths)
    )

    observations.extend(
        _run_h9(engine, strengths)
    )

    return FaultStrengthValidation(
        observations=tuple(observations)
    )