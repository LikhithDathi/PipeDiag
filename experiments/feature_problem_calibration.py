
from __future__ import annotations

import math

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)

from experiments.feature_problem_evidence import (
    collect_feature_problem_evidence,
)
from experiments.feature_problem_pipeline import (
    run_feature_problem_pipeline,
)


def collect_feature_problem_calibration_observations(
    n_samples: int = 500,
    n_features: int = 50,
    random_states: tuple[int, ...] = tuple(range(1, 21)),
    normal_constant_feature_ratios: tuple[float, ...] | None = None,
    faulty_constant_feature_ratios: tuple[float, ...] | None = None,
) -> list[CalibrationObservation]:

    if not random_states:
        raise ValueError("random_states must not be empty")

    if len(random_states) != len(set(random_states)):
        raise ValueError("random_states must be unique")

    if normal_constant_feature_ratios is None:
        normal_constant_feature_ratios = (
            0.00, 0.04, 0.08, 0.12, 0.16,
        )
    elif len(normal_constant_feature_ratios) != len(random_states):
        raise ValueError(
            "normal_constant_feature_ratios length must match random_states"
        )

    if faulty_constant_feature_ratios is None:
        faulty_constant_feature_ratios = (
            0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
        )
    elif len(faulty_constant_feature_ratios) != len(random_states):
        raise ValueError(
            "faulty_constant_feature_ratios length must match random_states"
        )

    for ratio in (
        normal_constant_feature_ratios
        + faulty_constant_feature_ratios
    ):
        if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
            raise ValueError("Invalid constant feature ratio")

    observations: list[CalibrationObservation] = []

    for random_state in random_states:

        # Normal scenarios: H8 is false.
        for ratio in normal_constant_feature_ratios:

            result = run_feature_problem_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                constant_feature_ratio=ratio,
                random_state=random_state,
            )

            evidence = collect_feature_problem_evidence(result)

            observations.append(
                CalibrationObservation(
                    evidence_group="G7",
                    hypothesis="H8",
                    hypothesis_true=False,
                    value=evidence.value,
                    measurement="constant_feature_ratio",
                )
            )

        # Faulty scenarios: H8 is true.
        for ratio in faulty_constant_feature_ratios:

            result = run_feature_problem_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                constant_feature_ratio=ratio,
                random_state=random_state,
            )

            evidence = collect_feature_problem_evidence(result)

            observations.append(
                CalibrationObservation(
                    evidence_group="G7",
                    hypothesis="H8",
                    hypothesis_true=True,
                    value=evidence.value,
                    measurement="constant_feature_ratio",
                )
            )

    return observations
