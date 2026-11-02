import numpy as np

from experiments.clean_pipeline import (
    CleanPipelineResult,
)
from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
)


def collect_clean_pipeline_evidence(
    result: CleanPipelineResult,
) -> list[EvidenceGroup]:

    if result is None:
        raise ValueError(
            "Pipeline result cannot be None"
        )

    return [
        EvidenceGroup(
            name="G2",
            value=float(result.accuracy),
            status="observed",
            metadata={
                "metric": "accuracy",
                "source": "clean_pipeline",
            },
        ),
        EvidenceGroup(
            name="G4",
            value=_class_balance_ratio(
                result.y_train
            ),
            status="observed",
            metadata={
                "metric": "class_balance_ratio",
                "source": "training_target",
            },
        ),
        EvidenceGroup(
            name="G5",
            value=_missing_value_rate(
                result.X_train
            ),
            status="observed",
            metadata={
                "metric": "missing_value_rate",
                "source": "training_features",
            },
        ),
        EvidenceGroup(
            name="G7",
            value=float(
                result.X_train.shape[1]
            ),
            status="observed",
            metadata={
                "metric": "feature_count",
                "source": "training_features",
            },
        ),
    ]


def _class_balance_ratio(
    target: np.ndarray,
) -> float:
    counts = np.bincount(target)

    if len(counts) < 2:
        return 1.0

    non_zero = counts[counts > 0]

    if len(non_zero) < 2:
        return 0.0

    return float(
        min(non_zero)
        / max(non_zero)
    )


def _missing_value_rate(
    features: np.ndarray,
) -> float:
    if features.size == 0:
        return 0.0

    return float(
        np.isnan(features).mean()
    )