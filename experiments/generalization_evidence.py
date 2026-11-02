
from __future__ import annotations

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_measurement import EvidenceMeasurement
from experiments.overfitting_pipeline import OverfittingPipelineResult


def collect_generalization_evidence(
    result: OverfittingPipelineResult,
) -> EvidenceGroup:
    if result is None:
        raise ValueError("result is required")

    generalization_gap = EvidenceMeasurement(
        name="generalization_gap",
        value=float(result.generalization_gap),
    )

    training_accuracy = EvidenceMeasurement(
        name="training_accuracy",
        value=float(result.train_accuracy),
    )

    evaluation_accuracy = EvidenceMeasurement(
        name="evaluation_accuracy",
        value=float(result.test_accuracy),
    )

    return EvidenceGroup(
        name="G2",
        value=generalization_gap.value,
        status="observed",
        metadata={
            "metric": "generalization_gap",
            "training_accuracy": training_accuracy.value,
            "evaluation_accuracy": evaluation_accuracy.value,
            "source": "overfitting_pipeline",
            "measurements": {
                generalization_gap.name: generalization_gap,
                training_accuracy.name: training_accuracy,
                evaluation_accuracy.name: evaluation_accuracy,
            },
        },
    )
