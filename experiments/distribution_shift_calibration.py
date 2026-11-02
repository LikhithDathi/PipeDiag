from __future__ import annotations

from pipediag.diagnostics.calibration import CalibrationObservation

from experiments.distribution_shift import (
    run_distribution_shift_pipeline,
)
from experiments.distribution_shift_evidence import (
    collect_distribution_shift_evidence,
)


def collect_distribution_shift_calibration_observations(
    n_samples: int = 500,
    n_features: int = 10,
    random_states: tuple[int, ...] = tuple(range(20)),
    normal_shift_magnitudes: tuple[float, ...] | None = None,
    shifted_magnitudes: tuple[float, ...] | None = None,
) -> list[CalibrationObservation]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not random_states:
        raise ValueError("random_states must not be empty")

    if normal_shift_magnitudes is None:
        normal_shift_magnitudes = tuple(
            0.0 for _ in random_states
        )

    if shifted_magnitudes is None:
        severity_levels = (
            0.50,
            0.75,
            1.00,
            1.25,
            1.50,
        )

        shifted_magnitudes = tuple(
            severity_levels[
                index % len(severity_levels)
            ]
            for index in range(len(random_states))
        )

    if len(normal_shift_magnitudes) != len(random_states):
        raise ValueError(
            "normal_shift_magnitudes must have the same length "
            "as random_states"
        )

    if len(shifted_magnitudes) != len(random_states):
        raise ValueError(
            "shifted_magnitudes must have the same length "
            "as random_states"
        )

    if any(
        magnitude < 0.0
        for magnitude in normal_shift_magnitudes
    ):
        raise ValueError(
            "normal_shift_magnitudes must be non-negative"
        )

    if any(
        magnitude <= 0.0
        for magnitude in shifted_magnitudes
    ):
        raise ValueError(
            "shifted_magnitudes must be positive"
        )

    observations: list[CalibrationObservation] = []

    for (
        random_state,
        normal_magnitude,
        shifted_magnitude,
    ) in zip(
        random_states,
        normal_shift_magnitudes,
        shifted_magnitudes,
    ):
        normal_result = run_distribution_shift_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            shift_magnitude=normal_magnitude,
            random_state=random_state,
        )

        normal_evidence = collect_distribution_shift_evidence(
            normal_result
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G3",
                hypothesis="H3",
                hypothesis_true=False,
                value=float(normal_evidence.value),
                measurement="mean_absolute_standardized_shift",
            )
        )

        shifted_result = run_distribution_shift_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            shift_magnitude=shifted_magnitude,
            random_state=random_state,
        )

        shifted_evidence = collect_distribution_shift_evidence(
            shifted_result
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G3",
                hypothesis="H3",
                hypothesis_true=True,
                value=float(shifted_evidence.value),
                measurement="mean_absolute_standardized_shift",
            )
        )

    return observations