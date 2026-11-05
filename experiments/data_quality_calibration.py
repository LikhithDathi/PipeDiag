
from __future__ import annotations

import math

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
    random_states: tuple[int, ...] = tuple(range(1, 21)),
    normal_missing_rates: tuple[float, ...] | None = None,
    faulty_missing_rates: tuple[float, ...] | None = None,
) -> list[CalibrationObservation]:

    if not random_states:
        raise ValueError("random_states must not be empty")

    if len(random_states) != len(set(random_states)):
        raise ValueError("random_states must be unique")

    if normal_missing_rates is None:
        normal_missing_rates = (
            0.00, 0.01, 0.02, 0.04, 0.06, 0.08,
        )
    elif len(normal_missing_rates) != len(random_states):
        raise ValueError(
            "normal_missing_rates length must match random_states"
        )

    if faulty_missing_rates is None:
        faulty_missing_rates = (
            0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
        )
    elif len(faulty_missing_rates) != len(random_states):
        raise ValueError(
            "faulty_missing_rates length must match random_states"
        )

    for rate in normal_missing_rates:
        if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
            raise ValueError("Invalid normal missing rate")

    for rate in faulty_missing_rates:
        if not math.isfinite(rate) or not 0.0 < rate < 1.0:
            raise ValueError("Invalid faulty missing rate")

    observations: list[CalibrationObservation] = []

    for random_state in random_states:

        # Normal scenarios: H5 is false.
        for missing_rate in normal_missing_rates:

            result = run_data_quality_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                missing_rate=missing_rate,
                random_state=random_state,
            )

            evidence = collect_data_quality_evidence(result)

            for hypothesis in ("H5", "H7"):
                observations.append(
                    CalibrationObservation(
                        evidence_group="G5",
                        hypothesis=hypothesis,
                        hypothesis_true=False,
                        value=evidence.value,
                        measurement="missing_value_rate",
                    )
                )

        # Faulty scenarios: H5 is true.
        for missing_rate in faulty_missing_rates:

            result = run_data_quality_pipeline(
                n_samples=n_samples,
                n_features=n_features,
                missing_rate=missing_rate,
                random_state=random_state,
            )

            evidence = collect_data_quality_evidence(result)

            for hypothesis in ("H5", "H7"):
                observations.append(
                    CalibrationObservation(
                        evidence_group="G5",
                        hypothesis=hypothesis,
                        hypothesis_true=True,
                        value=evidence.value,
                        measurement="missing_value_rate",
                    )
                )

    return observations
