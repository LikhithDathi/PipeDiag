from __future__ import annotations

from pipediag.diagnostics.calibration import CalibrationObservation

from experiments.imbalance_evidence import collect_imbalance_evidence
from experiments.imbalance_pipeline import run_imbalance_pipeline


def collect_imbalance_calibration_observations(
    n_samples: int = 500,
    n_features: int = 10,
    random_states: tuple[int, ...] = (1, 2, 3, 4, 5),
    normal_minority_fractions: tuple[float, ...] = (
        0.35,
        0.40,
        0.45,
        0.35,
        0.40,
    ),
    imbalanced_minority_fractions: tuple[float, ...] = (
        0.10,
        0.15,
        0.20,
        0.10,
        0.15,
    ),
) -> list[CalibrationObservation]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not random_states:
        raise ValueError("random_states must not be empty")

    if len(normal_minority_fractions) != len(random_states):
        raise ValueError(
            "normal_minority_fractions must have the same length "
            "as random_states"
        )

    if len(imbalanced_minority_fractions) != len(random_states):
        raise ValueError(
            "imbalanced_minority_fractions must have the same length "
            "as random_states"
        )

    observations: list[CalibrationObservation] = []

    for (
        random_state,
        normal_fraction,
        imbalanced_fraction,
    ) in zip(
        random_states,
        normal_minority_fractions,
        imbalanced_minority_fractions,
    ):
        normal_result = run_imbalance_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            minority_fraction=normal_fraction,
            random_state=random_state,
        )

        normal_evidence = collect_imbalance_evidence(normal_result)

        observations.append(
            CalibrationObservation(
                evidence_group="G4",
                hypothesis="H4",
                hypothesis_true=False,
                value=float(normal_evidence.value),
                measurement="minority_class_proportion",
            )
        )

        imbalanced_result = run_imbalance_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            minority_fraction=imbalanced_fraction,
            random_state=random_state,
        )

        imbalanced_evidence = collect_imbalance_evidence(
            imbalanced_result
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G4",
                hypothesis="H4",
                hypothesis_true=True,
                value=float(imbalanced_evidence.value),
                measurement="minority_class_proportion",
            )
        )

    return observations