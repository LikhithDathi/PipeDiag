from __future__ import annotations

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_measurement import (
    EvidenceMeasurement,
)

from experiments.feature_problem_pipeline import (
    FeatureProblemPipelineResult,
)


def collect_feature_problem_evidence(
    result: FeatureProblemPipelineResult,
) -> EvidenceGroup:

    if result is None:
        raise ValueError("result is required")

    constant_feature_ratio = EvidenceMeasurement(
        name="constant_feature_ratio",
        value=float(result.constant_feature_ratio),
    )

    return EvidenceGroup(
        name="G7",
        value=constant_feature_ratio.value,
        status="observed",
        metadata={
            "metric": "constant_feature_ratio",
            "constant_feature_ratio": (
                constant_feature_ratio.value
            ),
            "constant_feature_count": int(
                result.constant_feature_count
            ),
            "n_features": int(result.n_features),
            "accuracy": float(result.accuracy),
            "source": "feature_problem_pipeline",
            "measurements": {
                constant_feature_ratio.name:
                    constant_feature_ratio,
            },
        },
    )