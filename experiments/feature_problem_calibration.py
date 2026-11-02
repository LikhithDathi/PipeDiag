from __future__ import annotations

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
    n_features: int = 10,
    random_states: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
    ),
    normal_constant_feature_ratios: tuple[float, ...] = (
        0.00,
        0.00,
        0.10,
        0.00,
        0.10,
    ),
    faulty_constant_feature_ratios: tuple[float, ...] = (
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ),
) -> list[CalibrationObservation]:

    if not random_states:
        raise ValueError(
            "random_states must not be empty"
        )

    if len(normal_constant_feature_ratios) != len(
        random_states
    ):
        raise ValueError(
            "normal_constant_feature_ratios must have "
            "the same length as random_states"
        )

    if len(faulty_constant_feature_ratios) != len(
        random_states
    ):
        raise ValueError(
            "faulty_constant_feature_ratios must have "
            "the same length as random_states"
        )

    if any(
        not 0.0 <= ratio < 1.0
        for ratio in normal_constant_feature_ratios
    ):
        raise ValueError(
            "normal_constant_feature_ratios must be "
            "between 0 and 1"
        )

    if any(
        not 0.0 <= ratio < 1.0
        for ratio in faulty_constant_feature_ratios
    ):
        raise ValueError(
            "faulty_constant_feature_ratios must be "
            "between 0 and 1"
        )

    observations: list[CalibrationObservation] = []

    for (
        random_state,
        normal_ratio,
        faulty_ratio,
    ) in zip(
        random_states,
        normal_constant_feature_ratios,
        faulty_constant_feature_ratios,
    ):
        normal_result = run_feature_problem_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            constant_feature_ratio=normal_ratio,
            random_state=random_state,
        )

        normal_evidence = (
            collect_feature_problem_evidence(
                normal_result
            )
        )

        # G7 -> H8 primary relationship.
        observations.append(
            CalibrationObservation(
                evidence_group="G7",
                hypothesis="H8",
                hypothesis_true=False,
                value=normal_evidence.value,
                measurement="constant_feature_ratio",
            )
        )

        faulty_result = run_feature_problem_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            constant_feature_ratio=faulty_ratio,
            random_state=random_state,
        )

        faulty_evidence = (
            collect_feature_problem_evidence(
                faulty_result
            )
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G7",
                hypothesis="H8",
                hypothesis_true=True,
                value=faulty_evidence.value,
                measurement="constant_feature_ratio",
            )
        )

        # G7 -> H7 secondary relationship.
        #
        # The same feature evidence can also support
        # underfitting, but with a separate calibrated
        # hypothesis relationship.
        observations.append(
            CalibrationObservation(
                evidence_group="G7",
                hypothesis="H7",
                hypothesis_true=False,
                value=normal_evidence.value,
                measurement="constant_feature_ratio",
            )
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G7",
                hypothesis="H7",
                hypothesis_true=True,
                value=faulty_evidence.value,
                measurement="constant_feature_ratio",
            )
        )

    return observations