from __future__ import annotations
import re

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pipediag.application import DiagnosticRunResult


def _default_storage_dir() -> Path:
    return Path.home() / ".pipediag" / "runs"



def _validate_run_id(run_id: str) -> str:
    """Validate a run ID before using it in a storage path."""
    if not isinstance(run_id, str):
        raise ValueError("run_id must be a string")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("run_id contains invalid characters")

    return run_id



def _json_value(value: Any) -> Any:
    """Convert supported runtime values into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]

    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_value(value.tolist())

    return str(value)


def _runtime_events(runtime_context: Any | None) -> list[dict]:
    if runtime_context is None:
        return []

    getter = getattr(runtime_context, "get_events", None)
    if not callable(getter):
        return []

    events = []

    for event in getter() or []:
        timestamp = getattr(event, "timestamp", None)

        events.append({
            "event_id": str(getattr(event, "event_id", "")),
            "run_id": str(getattr(event, "run_id", "")),
            "operation": str(getattr(event, "operation", "unknown")),
            "component": getattr(event, "component", None),
            "step_name": getattr(event, "step_name", None),
            "timestamp": _json_value(timestamp),
            "duration_ms": _json_value(
                getattr(event, "duration_ms", None)
            ),
            "input_dataset_refs": _json_value(
                getattr(event, "input_dataset_refs", []) or []
            ),
            "output_dataset_refs": _json_value(
                getattr(event, "output_dataset_refs", []) or []
            ),
            "status": str(getattr(event, "status", "unknown")),
            "parent_event_id": getattr(
                event, "parent_event_id", None
            ),
            "depth": int(getattr(event, "depth", 0) or 0),
            "metadata": _json_value(
                getattr(event, "metadata", {}) or {}
            ),
        })

    return events


def _evaluations(runtime_context: Any | None) -> list[dict]:
    if runtime_context is None:
        return []

    getter = getattr(runtime_context, "get_evaluations", None)
    if not callable(getter):
        return []

    evaluations = []

    for item in getter() or []:
        metric_value = getattr(item, "metric_value", None)

        if metric_value is not None:
            try:
                metric_value = float(metric_value)
                if not math.isfinite(metric_value):
                    metric_value = None
            except (TypeError, ValueError):
                metric_value = None

        evaluations.append({
            "metric_name": getattr(item, "metric_name", None),
            "metric_value": metric_value,
            "evaluation_context": getattr(
                item, "evaluation_context", None
            ),
            "feature_dataset_id": getattr(
                item, "feature_dataset_id", None
            ),
            "target_dataset_id": getattr(
                item, "target_dataset_id", None
            ),
            "source": getattr(item, "source", None),
            "metadata": _json_value(
                getattr(item, "metadata", {}) or {}
            ),
        })

    return evaluations


def _serialize_result(
    result: DiagnosticRunResult,
    runtime_context: Any | None,
    run_id: str,
    run_type: str = "diagnostic",
) -> dict:
    diagnostic = result.diagnostic_result
    explanation = result.explanation

    events = _runtime_events(runtime_context)
    evaluations = _evaluations(runtime_context)

    statuses = [
        str(event["status"]).lower()
        for event in events
    ]

    durations = [
        event["duration_ms"]
        for event in events
        if isinstance(event["duration_ms"], (int, float))
    ]

    dataset_ids = {
        dataset_id
        for event in events
        for dataset_id in (
            event["input_dataset_refs"]
            + event["output_dataset_refs"]
        )
    }

    hypotheses = []

    for hypothesis in explanation.hypotheses:
        hypotheses.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "title": hypothesis.title,
            "posterior": hypothesis.posterior,
            "evidence": [
                {
                    "evidence_group": item.evidence_group,
                    "metric": item.metric,
                    "value": item.value,
                    "interpretation": item.interpretation,
                }
                for item in hypothesis.evidence
            ],
            "explanation_points": list(
                hypothesis.explanation_points
            ),
            "recommendation": hypothesis.recommendation,
        })

    evidence = []

    for item in diagnostic.evidence:
        metadata = _json_value(
            getattr(item, "metadata", {}) or {}
        )

        evidence.append({
            "evidence_type": str(
                getattr(item, "name", "unknown")
            ),
            "source": (
                str(metadata.get("source", "diagnostic"))
                if isinstance(metadata, dict)
                else "diagnostic"
            ),
            "status": str(getattr(item, "status", "unknown")),
            "measurements": (
                metadata.get("measurements", {})
                if isinstance(metadata, dict)
                and isinstance(metadata.get("measurements", {}), dict)
                else {}
            ),
            "reason": (
                metadata.get("reason")
                if isinstance(metadata, dict)
                else None
            ),
            "metadata": metadata,
        })



    multi_fault = None

    if result.multi_fault_result is not None:
        mf = result.multi_fault_result

        multi_fault = {
            "posterior": dict(mf.posterior),
            "independent_posterior": dict(
                mf.independent_posterior
            ),
            "detected_hypotheses": sorted(
                mf.detected_hypotheses
            ),
            "evidence_supported_hypotheses": sorted(
                mf.evidence_supported_hypotheses
            ),
            "uncertainty": mf.uncertainty,
            "uncertainty_by_hypothesis": dict(
                mf.uncertainty_by_hypothesis
            ),
            "detection_threshold": mf.detection_threshold,
            "evidence_log_likelihood_ratios": (
                mf.evidence_log_likelihood_ratios
            ),
            "metadata": mf.metadata,
        }

    return {
        "run_id": run_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic": {
            "posterior": dict(diagnostic.posterior),
            "uncertainty": diagnostic.uncertainty,
            "metadata": _json_value(dict(diagnostic.metadata)),
             "multi_fault": _json_value(multi_fault),
        },
        "explanation": {
            "threshold": explanation.threshold,
            "uncertainty": explanation.uncertainty,
            "inconclusive": explanation.is_inconclusive,
            "overall_assessment": explanation.overall_assessment,
            "hypotheses": hypotheses,
        },
        "report": result.report,
        "runtime_summary": {
            "total_events": len(events),
            "successful_events": statuses.count("success"),
            "failed_events": statuses.count("failed"),
            "running_events": statuses.count("running"),
            "total_duration_ms": sum(durations),
            "dataset_reference_count": len(dataset_ids),
            "evaluation_count": len(evaluations),
        },
        "runtime_events": events,
        "evaluations": evaluations,
        "evidence": evidence,
        "run_type": run_type,
        "multi_fault": multi_fault,
    }


def save_run(
    result: DiagnosticRunResult,
    runtime_context: Any | None = None,
    *,
    storage_dir: str | Path | None = None,
    run_type: str = "diagnostic",
) -> str:
    """Persist a completed diagnostic run and return its run ID."""
    if result is None:
        raise ValueError("diagnostic result is required")

    metadata = result.diagnostic_result.metadata or {}
    metadata_run_id = metadata.get("run_id")

    context_run_id = (
        getattr(runtime_context, "run_id", None)
        if runtime_context is not None
        else None
    )

    if context_run_id and metadata_run_id:
        if context_run_id != metadata_run_id:
            raise ValueError(
                "runtime context run_id does not match diagnostic metadata"
            )

    run_id = (
        context_run_id
        or metadata_run_id
        or f"R-{uuid4().hex[:12]}"
    )

    run_id = _validate_run_id(run_id)

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    record = _serialize_result(
        result,
        runtime_context,
        run_id,
        run_type,
    )

    directory = (
        Path(storage_dir).expanduser()
        if storage_dir is not None
        else _default_storage_dir()
    )

    directory.mkdir(parents=True, exist_ok=True)

    destination = directory / f"{run_id}.json"
    temporary = directory / f".{run_id}.{uuid4().hex}.tmp"

    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                record,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return run_id


def list_runs(
    *,
    storage_dir: str | Path | None = None,
) -> list[dict]:
    """Return saved run summaries, newest first."""
    directory = (
        Path(storage_dir).expanduser()
        if storage_dir is not None
        else _default_storage_dir()
    )

    if not directory.exists():
        return []

    runs = []

    for path in directory.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)

            runs.append({
                "run_id": record.get("run_id", path.stem),
                "saved_at": record.get("saved_at"),
                "uncertainty": record.get(
                    "diagnostic", {}
                ).get("uncertainty"),
                "run_type": record.get("run_type", "diagnostic"),
                "total_events": record.get(
                    "runtime_summary", {}
                ).get("total_events", 0),
            })
        except (OSError, json.JSONDecodeError):
            continue

    return sorted(
        runs,
        key=lambda item: item.get("saved_at") or "",
        reverse=True,
    )


def load_run(
    run_id: str,
    *,
    storage_dir: str | Path | None = None,
) -> dict | None:
    """Load a saved JSON snapshot by run ID."""
    run_id = _validate_run_id(run_id)

    directory = (
        Path(storage_dir).expanduser()
        if storage_dir is not None
        else _default_storage_dir()
    )

    path = directory / f"{run_id}.json"

    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def delete_run(
    run_id: str,
    *,
    storage_dir: str | Path | None = None,
) -> bool:
    """Delete a saved run. Return False if it does not exist."""
    run_id = _validate_run_id(run_id)

    directory = (
        Path(storage_dir).expanduser()
        if storage_dir is not None
        else _default_storage_dir()
    )

    path = directory / f"{run_id}.json"

    if not path.is_file():
        return False

    path.unlink()
    return True
