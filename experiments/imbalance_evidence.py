from __future__ import annotations

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_measurement import EvidenceMeasurement

from experiments.imbalance_pipeline import ImbalancePipelineResult


def collect_imbalance_evidence(
    result: ImbalancePipelineResult,
) -> EvidenceGroup:
    if result is None:
        raise ValueError("result is required")

    minority_proportion = EvidenceMeasurement(
        name="minority_class_proportion",
        value=float(result.minority_class_proportion),
    )

    imbalance_ratio = EvidenceMeasurement(
        name="imbalance_ratio",
        value=float(result.imbalance_ratio),
    )

    return EvidenceGroup(
        name="G4",
        value=minority_proportion.value,
        status="observed",
        metadata={
            "metric": "minority_class_proportion",
            "minority_class": result.minority_class,
            "minority_class_proportion": minority_proportion.value,
            "imbalance_ratio": imbalance_ratio.value,
            "train_class_counts": dict(result.train_class_counts),
            "test_class_counts": dict(result.test_class_counts),
            "source": "imbalance_pipeline",
            "measurements": {
                minority_proportion.name: minority_proportion,
                imbalance_ratio.name: imbalance_ratio,
            },
        },
    )