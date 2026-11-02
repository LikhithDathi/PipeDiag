from __future__ import annotations

from experiments.execution_problem_pipeline import (
    run_execution_problem_pipeline,
)
from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)


def _smoothed_failure_rate(
    failed_execution_events: int,
    total_execution_events: int,
) -> float:
    if total_execution_events <= 0:
        raise ValueError(
            "total_execution_events must be positive"
        )

    if not 0 <= failed_execution_events <= total_execution_events:
        raise ValueError(
            "failed_execution_events must be between 0 "
            "and total_execution_events"
        )

    # Jeffreys-style continuity correction for a
    # binomial proportion. This prevents boundary
    # values of exactly 0 or 1 during calibration.
    return (
        failed_execution_events + 0.5
    ) / (
        total_execution_events + 1.0
    )


def collect_execution_problem_calibration_observations(
    *,
    n_samples: int = 500,
    n_features: int = 10,
    random_states: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
    ),
    failure_event_counts: tuple[int, ...] = (
        1,
        1,
        2,
        2,
        1,
    ),
    total_execution_event_counts: tuple[int, ...] = (
        3,
        4,
        5,
        6,
        7,
    ),
) -> list[CalibrationObservation]:
    if not random_states:
        raise ValueError(
            "random_states must not be empty"
        )

    if len(failure_event_counts) != len(random_states):
        raise ValueError(
            "failure_event_counts must have the same "
            "length as random_states"
        )

    if len(total_execution_event_counts) != len(random_states):
        raise ValueError(
            "total_execution_event_counts must have the same "
            "length as random_states"
        )

    if any(
        count <= 0
        for count in failure_event_counts
    ):
        raise ValueError(
            "failure_event_counts must contain only "
            "positive values"
        )

    if any(
        count <= 0
        for count in total_execution_event_counts
    ):
        raise ValueError(
            "total_execution_event_counts must contain only "
            "positive values"
        )

    for failure_count, total_count in zip(
        failure_event_counts,
        total_execution_event_counts,
    ):
        if failure_count >= total_count:
            raise ValueError(
                "Each failure_event_count must be smaller "
                "than its corresponding total execution "
                "event count"
            )

    observations: list[CalibrationObservation] = []

    for (
        random_state,
        failure_count,
        total_event_count,
    ) in zip(
        random_states,
        failure_event_counts,
        total_execution_event_counts,
    ):
        clean_result = run_execution_problem_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            inject_failure=False,
            total_execution_events=total_event_count,
        )

        clean_value = _smoothed_failure_rate(
            failed_execution_events=(
                clean_result.failed_execution_events
            ),
            total_execution_events=(
                clean_result.total_execution_events
            ),
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G8",
                hypothesis="H9",
                hypothesis_true=False,
                value=clean_value,
                measurement="execution_failure_rate",
            )
        )

        faulty_result = run_execution_problem_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            inject_failure=True,
            failure_event_count=failure_count,
            total_execution_events=total_event_count,
        )

        faulty_value = _smoothed_failure_rate(
            failed_execution_events=(
                faulty_result.failed_execution_events
            ),
            total_execution_events=(
                faulty_result.total_execution_events
            ),
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G8",
                hypothesis="H9",
                hypothesis_true=True,
                value=faulty_value,
                measurement="execution_failure_rate",
            )
        )

    return observations