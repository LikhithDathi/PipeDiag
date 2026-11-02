from __future__ import annotations

import numpy as np

from pipediag.diagnostics.calibration import CalibrationObservation

from .leakage_evidence import leakage_score
from .leakage_pipeline import run_leakage_pipeline


def _default_leakage_strengths(
    count: int,
) -> tuple[float, ...]:
    """
    Generate deterministic leakage strengths for calibration.

    The values remain strongly leaky while providing enough variation
    for empirical variance estimation.
    """
    base_strengths = (
        0.80,
        0.84,
        0.88,
        0.92,
        0.96,
    )

    return tuple(
        base_strengths[index % len(base_strengths)]
        for index in range(count)
    )


def _collect_non_leaky_control(
    *,
    n_samples: int,
    n_features: int,
    random_state: int,
) -> float:
    """
    Measure G1 on a feature with no target-derived information.

    The control feature is generated independently of the target and
    therefore represents the H1=false condition using the same type of
    feature-target correlation measurement as the positive control.
    """
    rng = np.random.default_rng(random_state)

    # Recreate the same dataset used by the leakage pipeline.
    from experiments.dataset import create_dataset

    dataset = create_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    control_feature = rng.integers(
        0,
        2,
        size=n_samples,
    ).astype(float)

    features = np.column_stack(
        [
            dataset.features,
            control_feature,
        ]
    )

    return leakage_score(
        features,
        dataset.target,
        leakage_feature_index=n_features,
    )


def collect_leakage_calibration_observations(
    n_samples: int = 500,
    n_features: int = 10,
    random_states: tuple[int, ...] = (1, 2, 3, 4, 5),
    leakage_strengths: tuple[float, ...] | None = None,
) -> list[CalibrationObservation]:
    """
    Collect empirical G1 leakage observations for H1 calibration.

    Negative controls contain an independently generated random feature.
    Positive controls contain a target-derived leakage feature with
    controlled strength.

    Ground truth is used only to label the resulting observations; it is
    not used by the leakage evidence detector.
    """
    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive"
        )

    if n_features <= 0:
        raise ValueError(
            "n_features must be positive"
        )

    if not random_states:
        raise ValueError(
            "random_states cannot be empty"
        )

    if leakage_strengths is None:
        leakage_strengths = _default_leakage_strengths(
            len(random_states)
        )
    elif not leakage_strengths:
        raise ValueError(
            "leakage_strengths cannot be empty"
        )
    elif len(leakage_strengths) != len(random_states):
        raise ValueError(
            "random_states and leakage_strengths must have "
            "the same length"
        )

    observations: list[CalibrationObservation] = []

    for random_state, leakage_strength in zip(
        random_states,
        leakage_strengths,
    ):
        clean_score = _collect_non_leaky_control(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G1",
                hypothesis="H1",
                hypothesis_true=False,
                value=float(clean_score),
            )
        )

        leakage_result = run_leakage_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            leakage_strength=leakage_strength,
        )

        leakage_evidence = leakage_score(
            leakage_result.X_train,
            leakage_result.y_train,
            leakage_feature_index=(
                leakage_result.leakage_feature_index
            ),
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G1",
                hypothesis="H1",
                hypothesis_true=True,
                value=float(leakage_evidence),
            )
        )

    return observations