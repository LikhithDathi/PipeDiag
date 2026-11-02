from __future__ import annotations

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_measurement import EvidenceMeasurement

from experiments.distribution_shift import DistributionShiftPipelineResult


def collect_distribution_shift_evidence(
    result: DistributionShiftPipelineResult,
) -> EvidenceGroup:
    if result is None:
        raise ValueError("result is required")

    shift_score = EvidenceMeasurement(
        name="mean_absolute_standardized_shift",
        value=float(result.shift_magnitude),
    )

    return EvidenceGroup(
        name="G3",
        value=shift_score.value,
        status="observed",
        metadata={
            "metric": "mean_absolute_standardized_shift",
            "shift_score": shift_score.value,
            "source": "distribution_shift_pipeline",
            "measurements": {
                shift_score.name: shift_score,
            },
        },
    )