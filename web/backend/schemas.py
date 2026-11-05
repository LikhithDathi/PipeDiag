from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceResponse(BaseModel):
    evidence_group: str
    metric: str
    value: float
    interpretation: str


class AcquiredEvidenceResponse(BaseModel):
    evidence_type: str
    source: str
    status: str
    measurements: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeEventResponse(BaseModel):
    event_id: str
    run_id: str
    operation: str
    component: str | None = None
    step_name: str | None = None
    timestamp: str | None = None
    duration_ms: float | None = None
    input_dataset_refs: list[str] = Field(default_factory=list)
    output_dataset_refs: list[str] = Field(default_factory=list)
    status: str
    parent_event_id: str | None = None
    depth: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResponse(BaseModel):
    metric_name: str | None = None
    metric_value: float | None = None
    evaluation_context: str | None = None
    feature_dataset_id: str | None = None
    target_dataset_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeSummaryResponse(BaseModel):
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    running_events: int = 0
    total_duration_ms: float = 0.0
    dataset_reference_count: int = 0
    evaluation_count: int = 0


class HypothesisResponse(BaseModel):
    hypothesis_id: str
    title: str
    posterior: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceResponse]
    explanation_points: list[str]
    recommendation: str
    evidence_supported: bool = True


class DiagnosticResponse(BaseModel):
    posterior: dict[str, float]
    uncertainty: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any]
    multi_fault: MultiFaultDiagnosticResponse | None = None


class ExplanationResponse(BaseModel):
    threshold: float
    uncertainty: float = Field(ge=0.0, le=1.0)
    inconclusive: bool
    overall_assessment: str
    hypotheses: list[HypothesisResponse]


class DiagnosticRunResponse(BaseModel):
    diagnostic: DiagnosticResponse
    explanation: ExplanationResponse
    report: str
    runtime_summary: RuntimeSummaryResponse = Field(
        default_factory=RuntimeSummaryResponse
    )
    runtime_events: list[RuntimeEventResponse] = Field(
        default_factory=list
    )
    evaluations: list[EvaluationResponse] = Field(
        default_factory=list
    )
    evidence: list[AcquiredEvidenceResponse] = Field(
        default_factory=list
    )
    run_type: str = "diagnostic"






class MultiFaultDiagnosticResponse(BaseModel):
    posterior: dict[str, float]
    independent_posterior: dict[str, float]
    detected_hypotheses: list[str]
    evidence_supported_hypotheses: list[str]
    uncertainty: float
    uncertainty_by_hypothesis: dict[str, float]
    detection_threshold: float
    evidence_log_likelihood_ratios: dict[
        str, dict[str, float]
    ] = {}
    metadata: dict[str, Any] = {}
