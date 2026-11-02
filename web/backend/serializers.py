from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pipediag.application import DiagnosticRunResult

from .schemas import (
    AcquiredEvidenceResponse,
    DiagnosticResponse,
    DiagnosticRunResponse,
    EvaluationResponse,
    EvidenceResponse,
    ExplanationResponse,
    HypothesisResponse,
    RuntimeEventResponse,
    RuntimeSummaryResponse,
)


def _to_json_value(value: Any) -> Any:
    """Convert common runtime values to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _to_json_value(value.tolist())
    return str(value)


def _serialize_runtime_events(runtime_context: Any | None) -> list[RuntimeEventResponse]:
    if runtime_context is None:
        return []

    get_events = getattr(runtime_context, "get_events", None)
    if not callable(get_events):
        return []

    events = get_events() or []
    serialized = []

    for event in events:
        timestamp = getattr(event, "timestamp", None)
        serialized.append(
            RuntimeEventResponse(
                event_id=str(getattr(event, "event_id", "")),
                run_id=str(getattr(event, "run_id", "")),
                operation=str(getattr(event, "operation", "unknown")),
                component=getattr(event, "component", None),
                step_name=getattr(event, "step_name", None),
                timestamp=(
                    timestamp.isoformat()
                    if isinstance(timestamp, (datetime, date))
                    else str(timestamp) if timestamp is not None else None
                ),
                duration_ms=getattr(event, "duration_ms", None),
                input_dataset_refs=list(
                    getattr(event, "input_dataset_refs", []) or []
                ),
                output_dataset_refs=list(
                    getattr(event, "output_dataset_refs", []) or []
                ),
                status=str(getattr(event, "status", "unknown")),
                parent_event_id=getattr(event, "parent_event_id", None),
                depth=int(getattr(event, "depth", 0) or 0),
                metadata=_to_json_value(
                    getattr(event, "metadata", {}) or {}
                ),
            )
        )

    return serialized


def _serialize_evaluations(runtime_context: Any | None) -> list[EvaluationResponse]:
    if runtime_context is None:
        return []

    get_evaluations = getattr(runtime_context, "get_evaluations", None)
    if not callable(get_evaluations):
        return []

    serialized = []
    for evaluation in get_evaluations() or []:
        metric_value = getattr(evaluation, "metric_value", None)
        try:
            metric_value = (
                float(metric_value) if metric_value is not None else None
            )
        except (TypeError, ValueError):
            metric_value = None

        serialized.append(
            EvaluationResponse(
                metric_name=getattr(evaluation, "metric_name", None),
                metric_value=metric_value,
                evaluation_context=getattr(
                    evaluation, "evaluation_context", None
                ),
                feature_dataset_id=getattr(
                    evaluation, "feature_dataset_id", None
                ),
                target_dataset_id=getattr(
                    evaluation, "target_dataset_id", None
                ),
                source=getattr(evaluation, "source", None),
                metadata=_to_json_value(
                    getattr(evaluation, "metadata", {}) or {}
                ),
            )
        )

    return serialized


def _serialize_acquired_evidence(diagnostic_result: Any) -> list[AcquiredEvidenceResponse]:
    serialized = []

    for item in getattr(diagnostic_result, "evidence", []) or []:
        group_name = getattr(item, "name", None)
        metadata = _to_json_value(getattr(item, "metadata", {}) or {})
        measurements = metadata.get("measurements", {}) if isinstance(metadata, dict) else {}

        serialized.append(
            AcquiredEvidenceResponse(
                evidence_type=str(group_name or "unknown"),
                source=str(metadata.get("source", "diagnostic") if isinstance(metadata, dict) else "diagnostic"),
                status=str(getattr(item, "status", "unknown")),
                measurements=measurements if isinstance(measurements, dict) else {},
                reason=metadata.get("reason") if isinstance(metadata, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )

    return serialized


def diagnostic_run_result_to_response(
    result: DiagnosticRunResult,
    runtime_context: Any | None = None,
    *,
    run_type: str = "diagnostic",
) -> DiagnosticRunResponse:
    """Serialize a completed run without changing its diagnostic result."""
    if result is None:
        raise ValueError("diagnostic result is required")

    diagnostic_result = result.diagnostic_result
    explanation = result.explanation

    hypotheses = []
    for hypothesis in explanation.hypotheses:
        evidence = [
            EvidenceResponse(
                evidence_group=item.evidence_group,
                metric=item.metric,
                value=item.value,
                interpretation=item.interpretation,
            )
            for item in hypothesis.evidence
        ]

        hypotheses.append(
            HypothesisResponse(
                hypothesis_id=hypothesis.hypothesis_id,
                title=hypothesis.title,
                posterior=hypothesis.posterior,
                evidence=evidence,
                explanation_points=list(hypothesis.explanation_points),
                recommendation=hypothesis.recommendation,
            )
        )

    runtime_events = _serialize_runtime_events(runtime_context)
    evaluations = _serialize_evaluations(runtime_context)

    event_statuses = [
        str(getattr(event, "status", "unknown")).lower()
        for event in (getattr(runtime_context, "get_events", lambda: [])() or [])
    ] if runtime_context is not None else []

    durations = [
        float(event.duration_ms)
        for event in runtime_events
        if event.duration_ms is not None
    ]

    dataset_ids = {
        dataset_id
        for event in runtime_events
        for dataset_id in (
            event.input_dataset_refs + event.output_dataset_refs
        )
    }

    evidence = _serialize_acquired_evidence(diagnostic_result)

    return DiagnosticRunResponse(
        diagnostic=DiagnosticResponse(
            posterior=dict(diagnostic_result.posterior),
            uncertainty=diagnostic_result.uncertainty,
            metadata=_to_json_value(dict(diagnostic_result.metadata)),
        ),
        explanation=ExplanationResponse(
            threshold=explanation.threshold,
            uncertainty=explanation.uncertainty,
            inconclusive=explanation.is_inconclusive,
            overall_assessment=explanation.overall_assessment,
            hypotheses=hypotheses,
        ),
        report=result.report,
        runtime_summary=RuntimeSummaryResponse(
            total_events=len(runtime_events),
            successful_events=sum(status == "success" for status in event_statuses),
            failed_events=sum(status == "failed" for status in event_statuses),
            running_events=sum(status == "running" for status in event_statuses),
            total_duration_ms=sum(durations),
            dataset_reference_count=len(dataset_ids),
            evaluation_count=len(evaluations),
        ),
        runtime_events=runtime_events,
        evaluations=evaluations,
        evidence=evidence,
        run_type=run_type,
    )
