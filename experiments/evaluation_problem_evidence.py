
from __future__ import annotations

import math

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

    wilson_half_width = EvidenceMeasurement(
        name="wilson_interval_half_width",
        value=float(result.wilson_interval_half_width),
    )

    sample_count = EvidenceMeasurement(
        name="evaluation_sample_count",
        value=float(result.evaluation_sample_count),
    )

    return EvidenceGroup(
        name="G6",
        value=wilson_half_width.value,
        status="observed",
        metadata={
            "metric": "wilson_interval_half_width",
            "wilson_interval_half_width": wilson_half_width.value,

            "evaluation_sample_count": int(
                result.evaluation_sample_count
            ),
            "accuracy": float(result.accuracy),
            "max_wilson_half_width": float(
                result.max_wilson_half_width
            ),
            "evaluation_precision_adequate": bool(
                result.evaluation_precision_adequate
            ),
            "source": "evaluation_problem_pipeline",
            "measurements": {
                wilson_half_width.name: wilson_half_width,
                sample_count.name: sample_count,

                "log_wilson_interval_half_width": EvidenceMeasurement(
                    name="log_wilson_interval_half_width",
                    value=math.log(result.wilson_interval_half_width),
                ),
            },
        },
    )
