
from __future__ import annotations

import math

from pipediag.diagnostics.calibration import CalibrationObservation

from experiments.evaluation_problem_pipeline import (
    DEFAULT_MAX_WILSON_HALF_WIDTH,
    run_evaluation_problem_pipeline,
)


def collect_evaluation_problem_calibration_observations(
    n_features: int = 10,
    n_samples_values: tuple[int, ...] = (
        200, 300, 500, 750, 1000, 1500, 2000,
    ),
    test_sizes: tuple[float, ...] = (
        0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
    ),
    random_states: tuple[int, ...] = tuple(range(1, 21)),
    max_wilson_half_width: float = DEFAULT_MAX_WILSON_HALF_WIDTH,
) -> list[CalibrationObservation]:

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not n_samples_values or any(n <= 0 for n in n_samples_values):
        raise ValueError("n_samples_values must contain positive values")

    if not test_sizes or any(not 0.0 < size < 1.0 for size in test_sizes):
        raise ValueError("test_sizes must be between 0 and 1")

    if not random_states:
        raise ValueError("random_states must not be empty")

    if not 0.0 < max_wilson_half_width < 1.0:
        raise ValueError("max_wilson_half_width must be between 0 and 1")

    observations: list[CalibrationObservation] = []

    for n_samples in n_samples_values:
        for test_size in test_sizes:
            for random_state in random_states:
                result = run_evaluation_problem_pipeline(
                    n_samples=n_samples,
                    n_features=n_features,
                    test_size=test_size,
                    random_state=random_state,
                    max_wilson_half_width=max_wilson_half_width,
                )

                observations.append(
                    CalibrationObservation(
                        evidence_group="G6",
                        hypothesis="H6",
                        hypothesis_true=(
                            not result.evaluation_precision_adequate
                        ),
                        value=math.log(result.wilson_interval_half_width),
                        measurement="log_wilson_interval_half_width",
                    )
                )

    true_count = sum(
        observation.hypothesis_true
        for observation in observations
    )
    false_count = len(observations) - true_count

    if true_count == 0 or false_count == 0:
        raise ValueError(
            "G6 calibration must contain both H6 TRUE and H6 FALSE "
            "observations. Expand the sample-size or test-size ranges."
        )

    return observations
