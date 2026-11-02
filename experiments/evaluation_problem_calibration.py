from __future__ import annotations

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)

from experiments.evaluation_problem_evidence import (
    collect_evaluation_problem_evidence,
)
from experiments.evaluation_problem_pipeline import (
    run_evaluation_problem_pipeline,
)


def collect_evaluation_problem_calibration_observations(
    n_samples: int = 500,
    n_features: int = 10,
    random_states: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
    ),
    reliable_test_sizes: tuple[float, ...] = (
        0.20,
        0.25,
        0.20,
        0.25,
        0.20,
    ),
    unreliable_test_sizes: tuple[float, ...] = (
        0.02,
        0.03,
        0.04,
        0.05,
        0.02,
    ),
) -> list[CalibrationObservation]:

    if not random_states:
        raise ValueError(
            "random_states must not be empty"
        )

    if len(reliable_test_sizes) != len(
        random_states
    ):
        raise ValueError(
            "reliable_test_sizes must have the "
            "same length as random_states"
        )

    if len(unreliable_test_sizes) != len(
        random_states
    ):
        raise ValueError(
            "unreliable_test_sizes must have the "
            "same length as random_states"
        )

    if any(
        not 0.0 < size < 1.0
        for size in reliable_test_sizes
    ):
        raise ValueError(
            "reliable_test_sizes must be between "
            "0 and 1"
        )

    if any(
        not 0.0 < size < 1.0
        for size in unreliable_test_sizes
    ):
        raise ValueError(
            "unreliable_test_sizes must be between "
            "0 and 1"
        )

    observations: list[
        CalibrationObservation
    ] = []

    for (
        random_state,
        reliable_size,
        unreliable_size,
    ) in zip(
        random_states,
        reliable_test_sizes,
        unreliable_test_sizes,
    ):

        reliable_result = (
            run_evaluation_problem_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                test_size=reliable_size,
                random_state=random_state,
            )
        )

        reliable_evidence = (
            collect_evaluation_problem_evidence(
                reliable_result
            )
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G6",
                hypothesis="H6",
                hypothesis_true=False,
                value=reliable_evidence.value,
                measurement="evaluation_standard_error",
            )
        )

        unreliable_result = (
            run_evaluation_problem_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                test_size=unreliable_size,
                random_state=random_state,
            )
        )

        unreliable_evidence = (
            collect_evaluation_problem_evidence(
                unreliable_result
            )
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G6",
                hypothesis="H6",
                hypothesis_true=True,
                value=unreliable_evidence.value,
                measurement="evaluation_standard_error",
            )
        )

    return observations