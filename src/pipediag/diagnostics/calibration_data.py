from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
)


def build_calibration_observations(
    evidence_groups: list[EvidenceGroup],
    expected_hypothesis: str | None,
    hypotheses: list[str],
) -> list[CalibrationObservation]:

    if not evidence_groups:
        raise ValueError(
            "Evidence groups cannot be empty"
        )

    if not hypotheses:
        raise ValueError(
            "Hypotheses cannot be empty"
        )

    if len(hypotheses) != len(set(hypotheses)):
        raise ValueError(
            "Hypotheses must be unique"
        )

    if expected_hypothesis is not None:
        if expected_hypothesis not in hypotheses:
            raise ValueError(
                "Expected hypothesis must belong to "
                "the configured hypotheses"
            )

    evidence_names = [
        evidence.name
        for evidence in evidence_groups
    ]

    if len(evidence_names) != len(
        set(evidence_names)
    ):
        raise ValueError(
            "Evidence group names must be unique"
        )

    observations = []

    for evidence in evidence_groups:
        if evidence.status == "unavailable":
            continue

        if evidence.status != "observed":
            raise ValueError(
                "Evidence status must be either "
                "'observed' or 'unavailable'"
            )

        for hypothesis in hypotheses:
            is_true = (
                expected_hypothesis is not None
                and hypothesis == expected_hypothesis
            )

            observations.append(
                CalibrationObservation(
                    evidence_group=evidence.name,
                    hypothesis=hypothesis,
                    hypothesis_true=is_true,
                    value=float(evidence.value),
                )
            )

    if not observations:
        return []

    return observations