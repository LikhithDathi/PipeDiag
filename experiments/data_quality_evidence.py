from __future__ import annotations

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_measurement import (
    EvidenceMeasurement,
)

from experiments.data_quality_pipeline import (
    DataQualityPipelineResult,
)


def collect_data_quality_evidence(
    result: DataQualityPipelineResult,
) -> EvidenceGroup:

    if result is None:
        raise ValueError("result is required")

    missing_value_rate = EvidenceMeasurement(
        name="missing_value_rate",
        value=float(result.missing_value_rate),
    )

    return EvidenceGroup(
        name="G5",
        value=missing_value_rate.value,
        status="observed",
        metadata={
            "metric": "missing_value_rate",
            "missing_value_rate": missing_value_rate.value,
            "source": "data_quality_pipeline",
            "accuracy": float(result.accuracy),
            "measurements": {
                missing_value_rate.name:
                    missing_value_rate,
            },
        },
    )