from __future__ import annotations

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)

from experiments.data_quality_evidence import (
    collect_data_quality_evidence,
)
from experiments.data_quality_pipeline import (
    run_data_quality_pipeline,
)


def collect_data_quality_calibration_observations(
    n_samples: int = 500,
    n_features: int = 10,
    random_states: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
    ),
    normal_missing_rates: tuple[float, ...] = (
        0.00,
        0.01,
        0.02,
        0.00,
        0.01,
    ),
    faulty_missing_rates: tuple[float, ...] = (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    ),
) -> list[CalibrationObservation]:

    if not random_states:
        raise ValueError(
            "random_states must not be empty"
        )

    if len(normal_missing_rates) != len(
        random_states
    ):
        raise ValueError(
            "normal_missing_rates must have the "
            "same length as random_states"
        )

    if len(faulty_missing_rates) != len(
        random_states
    ):
        raise ValueError(
            "faulty_missing_rates must have the "
            "same length as random_states"
        )

    if any(
        rate < 0.0
        for rate in normal_missing_rates
    ):
        raise ValueError(
            "normal_missing_rates must be non-negative"
        )

    if any(
        rate <= 0.0
        for rate in faulty_missing_rates
    ):
        raise ValueError(
            "faulty_missing_rates must be positive"
        )

    observations: list[
        CalibrationObservation
    ] = []

    for (
        random_state,
        normal_rate,
        faulty_rate,
    ) in zip(
        random_states,
        normal_missing_rates,
        faulty_missing_rates,
    ):

        # ----------------------------------------------------
        # Normal / non-H5 scenario
        # ----------------------------------------------------

        normal_result = run_data_quality_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            missing_rate=normal_rate,
            random_state=random_state,
        )

        normal_evidence = (
            collect_data_quality_evidence(
                normal_result
            )
        )

        # G5 -> H5 primary
        observations.append(
            CalibrationObservation(
                evidence_group="G5",
                hypothesis="H5",
                hypothesis_true=False,
                value=normal_evidence.value,
                measurement="missing_value_rate",
            )
        )

        # G5 -> H7 secondary
        observations.append(
            CalibrationObservation(
                evidence_group="G5",
                hypothesis="H7",
                hypothesis_true=False,
                value=normal_evidence.value,
                measurement="missing_value_rate",
            )
        )

        # ----------------------------------------------------
        # Faulty / H5 scenario
        # ----------------------------------------------------

        faulty_result = run_data_quality_pipeline(
            n_samples=n_samples,
            n_features=n_features,
            missing_rate=faulty_rate,
            random_state=random_state,
        )

        faulty_evidence = (
            collect_data_quality_evidence(
                faulty_result
            )
        )

        # G5 -> H5 primary
        observations.append(
            CalibrationObservation(
                evidence_group="G5",
                hypothesis="H5",
                hypothesis_true=True,
                value=faulty_evidence.value,
                measurement="missing_value_rate",
            )
        )

        # G5 -> H7 secondary
        observations.append(
            CalibrationObservation(
                evidence_group="G5",
                hypothesis="H7",
                hypothesis_true=True,
                value=faulty_evidence.value,
                measurement="missing_value_rate",
            )
        )

    return observations