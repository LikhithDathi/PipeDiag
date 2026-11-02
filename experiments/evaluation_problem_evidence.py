from __future__ import annotations

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_measurement import (
    EvidenceMeasurement,
)

from experiments.evaluation_problem_pipeline import (
    EvaluationProblemPipelineResult,
)


def collect_evaluation_problem_evidence(
    result: EvaluationProblemPipelineResult,
) -> EvidenceGroup:

    if result is None:
        raise ValueError("result is required")

    evaluation_standard_error = EvidenceMeasurement(
        name="evaluation_standard_error",
        value=float(
            result.evaluation_standard_error
        ),
    )

    return EvidenceGroup(
        name="G6",
        value=evaluation_standard_error.value,
        status="observed",
        metadata={
            "metric": "evaluation_standard_error",
            "evaluation_standard_error":
                evaluation_standard_error.value,
            "evaluation_sample_count":
                int(result.evaluation_sample_count),
            "accuracy": float(result.accuracy),
            "source": "evaluation_problem_pipeline",
            "measurements": {
                evaluation_standard_error.name:
                    evaluation_standard_error,
            },
        },
    )