from __future__ import annotations

from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
)
from pipediag.diagnostics.evidence_measurement import (
    EvidenceMeasurement,
)

from experiments.execution_problem_pipeline import (
    ExecutionProblemPipelineResult,
)


def collect_execution_problem_evidence(
    result: ExecutionProblemPipelineResult,
) -> EvidenceGroup:

    if result is None:
        raise ValueError(
            "result is required"
        )

    execution_failure_rate = EvidenceMeasurement(
        name="execution_failure_rate",
        value=float(
            result.execution_failure_rate
        ),
    )

    return EvidenceGroup(
        name="G8",
        value=execution_failure_rate.value,
        status="observed",
        metadata={
            "metric": "execution_failure_rate",
            "execution_failure_rate": (
                execution_failure_rate.value
            ),
            "total_execution_events": int(
                result.total_execution_events
            ),
            "failed_execution_events": int(
                result.failed_execution_events
            ),
            "fault_injected": bool(
                result.fault_injected
            ),
            "accuracy": float(
                result.accuracy
            ),
            "source": (
                "execution_problem_pipeline"
            ),
            "measurements": {
                execution_failure_rate.name:
                    execution_failure_rate,
            },
        },
    )