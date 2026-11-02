from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.calibration_data import (
    build_calibration_observations,
)
from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)


class CalibrationRunner:
    def __init__(
        self,
        evidence_provider,
    ):
        if evidence_provider is None:
            raise ValueError(
                "CalibrationRunner requires an evidence provider"
            )

        self.evidence_provider = evidence_provider

    def collect(
        self,
        scenarios: list[DiagnosticScenario],
        hypotheses: list[str],
    ) -> list[CalibrationObservation]:

        if not scenarios:
            raise ValueError(
                "Calibration scenarios cannot be empty"
            )

        if not hypotheses:
            raise ValueError(
                "Calibration hypotheses cannot be empty"
            )

        if len(hypotheses) != len(
            set(hypotheses)
        ):
            raise ValueError(
                "Calibration hypotheses must be unique"
            )

        scenario_ids = [
            scenario.scenario_id
            for scenario in scenarios
        ]

        if len(scenario_ids) != len(
            set(scenario_ids)
        ):
            raise ValueError(
                "Calibration scenario IDs must be unique"
            )

        observations = []

        for scenario in scenarios:
            evidence = self.evidence_provider.collect(
                scenario
            )

            scenario_observations = (
                build_calibration_observations(
                    evidence_groups=evidence,
                    expected_hypothesis=(
                        scenario.expected_hypothesis
                    ),
                    hypotheses=hypotheses,
                )
            )

            observations.extend(
                scenario_observations
            )

        return observations