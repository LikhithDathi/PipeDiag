from __future__ import annotations

from pipediag.diagnostics.calibration import CalibrationObservation

from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import run_overfitting_pipeline


def collect_underfitting_calibration_observations(
    n_samples: int = 500,
    n_features: int = 10,
    random_states: tuple[int, ...] = tuple(range(1, 31)),
    normal_max_depths: tuple[int, ...] = (2, 3, 4, 5),
    underfitting_max_depth: int = 1,
) -> list[CalibrationObservation]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not random_states:
        raise ValueError("random_states must not be empty")

    if not normal_max_depths:
        raise ValueError("normal_max_depths must not be empty")

    if underfitting_max_depth <= 0:
        raise ValueError("underfitting_max_depth must be positive")

    if any(depth <= 0 for depth in normal_max_depths):
        raise ValueError(
            "normal_max_depths must contain only positive values"
        )

    if underfitting_max_depth in normal_max_depths:
        raise ValueError(
            "underfitting_max_depth must not be included "
            "in normal_max_depths"
        )

    observations: list[CalibrationObservation] = []

    for random_state in random_states:
        underfitting_result = run_overfitting_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            max_depth=underfitting_max_depth,
        )

        underfitting_evidence = collect_generalization_evidence(
            underfitting_result
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G2",
                hypothesis="H7",
                hypothesis_true=True,
                value=float(
                    underfitting_evidence.metadata["training_accuracy"]
                ),
                measurement="training_accuracy",
            )
        )

        for normal_max_depth in normal_max_depths:
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
                    hypothesis="H7",
                    hypothesis_true=False,
                    value=float(
                        normal_evidence.metadata["training_accuracy"]
                    ),
                    measurement="training_accuracy",
                )
            )

    return observations
