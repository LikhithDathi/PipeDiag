from dataclasses import dataclass


@dataclass(frozen=True)
class GroundTruth:
    scenario_id: str
    hypothesis: str
    is_true: bool


def create_ground_truth(
    scenario_id: str,
    expected_hypothesis: str | None,
    hypotheses: list[str],
) -> list[GroundTruth]:
    if not scenario_id:
        raise ValueError(
            "scenario_id must be non-empty"
        )

    if not hypotheses:
        raise ValueError(
            "At least one hypothesis is required"
        )

    if len(hypotheses) != len(set(hypotheses)):
        raise ValueError(
            "Hypotheses must be unique"
        )

    if expected_hypothesis is None:
        return []

    if expected_hypothesis not in hypotheses:
        raise ValueError(
            "Expected hypothesis must belong to "
            "the configured hypotheses"
        )

    return [
        GroundTruth(
            scenario_id=scenario_id,
            hypothesis=hypothesis,
            is_true=(
                hypothesis == expected_hypothesis
            ),
        )
        for hypothesis in hypotheses
    ]