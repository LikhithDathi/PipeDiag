
from __future__ import annotations

from pipediag.diagnostics.calibration import CalibrationObservation

from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import run_overfitting_pipeline


def collect_generalization_calibration_observations(
    n_samples: int = 500,
    n_features: int = 10,
    random_states: tuple[int, ...] = tuple(range(1, 31)),
    normal_max_depth: int = 2,
):
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not random_states:
        raise ValueError("random_states must not be empty")

    if normal_max_depth <= 0:
        raise ValueError("normal_max_depth must be positive")

    observations: list[CalibrationObservation] = []

    for random_state in random_states:
        normal_result = run_overfitting_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            max_depth=normal_max_depth,
        )

        normal_evidence = collect_generalization_evidence(
            normal_result
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G2",
                hypothesis="H2",
                hypothesis_true=False,
                value=float(normal_evidence.value),
                measurement="generalization_gap",
            )
        )

        overfit_result = run_overfitting_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            max_depth=None,
        )

        overfit_evidence = collect_generalization_evidence(
            overfit_result
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G2",
                hypothesis="H2",
                hypothesis_true=True,
                value=float(overfit_evidence.value),
                measurement="generalization_gap",
            )
        )

    return observations
