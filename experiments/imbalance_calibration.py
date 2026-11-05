
from __future__ import annotations

from itertools import product

from pipediag.diagnostics.calibration import CalibrationObservation

from experiments.imbalance_evidence import collect_imbalance_evidence
from experiments.imbalance_pipeline import run_imbalance_pipeline


def collect_imbalance_calibration_observations(
    n_samples: int = 1000,
    n_features: int = 20,
    random_states: tuple[int, ...] = tuple(range(1, 21)),
    normal_minority_fractions: tuple[float, ...] = (
        0.26,
        0.30,
        0.34,
        0.38,
        0.42,
        0.46,
        0.49,
    ),
    imbalanced_minority_fractions: tuple[float, ...] = (
        0.05,
        0.08,
        0.11,
        0.14,
        0.17,
        0.20,
        0.23,
    ),
) -> list[CalibrationObservation]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not random_states:
        raise ValueError("random_states must not be empty")

    if not normal_minority_fractions:
        raise ValueError("normal_minority_fractions must not be empty")

    if not imbalanced_minority_fractions:
        raise ValueError("imbalanced_minority_fractions must not be empty")

    if any(not 0.25 < value < 0.5 for value in normal_minority_fractions):
        raise ValueError(
            "Normal fractions must be greater than 0.25 and less than 0.5"
        )

    if any(not 0.0 < value < 0.25 for value in imbalanced_minority_fractions):
        raise ValueError(
            "Imbalanced fractions must be greater than 0 and less than 0.25"
        )

    observations: list[CalibrationObservation] = []

    scenarios = (
        (normal_minority_fractions, False),
        (imbalanced_minority_fractions, True),
    )

    for fractions, hypothesis_true in scenarios:
        for random_state, fraction in product(random_states, fractions):
            result = run_imbalance_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                minority_fraction=fraction,
                random_state=random_state,
            )

            evidence = collect_imbalance_evidence(result)

            observations.append(
                CalibrationObservation(
                    evidence_group="G4",
                    hypothesis="H4",
                    hypothesis_true=hypothesis_true,
                    value=float(evidence.value),
                    measurement="minority_class_proportion",
                )
            )

    return observations
