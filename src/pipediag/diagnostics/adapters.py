from __future__ import annotations
from .evidence_group import EvidenceGroup

from typing import Any

from .evidence import (
    DiagnosticEvidence,
    EVIDENCE_OBSERVED,
    EVIDENCE_INSUFFICIENT,
)


EVIDENCE_TYPE_TO_GROUP = {
    "leakage": "G1",
    "generalization": "G2",
    "distribution_shift": "G3",
    "imbalance": "G4",
    "data_quality": "G5",
    "evaluation_problem": "G6",
    "feature_problem": "G7",
    "execution_problem": "G8",
}


def _status_to_common(
    status: str,
) -> str:
    """
    Convert a diagnostic-specific status into the
    common evidence status.

    Any status that explicitly indicates unavailable
    evidence becomes INSUFFICIENT_EVIDENCE.
    Other statuses are preserved as observed evidence.
    """

    if status in {
        "INSUFFICIENT_EVIDENCE",
        "UNAVAILABLE",
    }:
        return EVIDENCE_INSUFFICIENT

    return EVIDENCE_OBSERVED


def _common_metadata(
    evidence: Any,
) -> dict[str, Any]:
    """
    Preserve diagnostic-specific metadata without
    changing the original evidence object.
    """

    metadata = getattr(
        evidence,
        "metadata",
        None,
    )

    if isinstance(metadata, dict):
        return dict(metadata)

    return {}


def generalization_to_evidence(
    evidence: Any,
) -> DiagnosticEvidence:
    """
    Convert GeneralizationEvidence into the common
    DiagnosticEvidence representation.
    """

    status = getattr(
        evidence,
        "status",
        "UNAVAILABLE",
    )

    measurements: dict[str, Any] = {}

    for field_name in (
        "train_value",
        "test_value",
        "gap",
    ):
        value = getattr(
            evidence,
            field_name,
            None,
        )

        if value is not None:
            measurements[field_name] = value

    metric_direction = getattr(
        evidence,
        "metric_direction",
        None,
    )

    if metric_direction is not None:
        measurements[
            "metric_direction"
        ] = metric_direction

    metric_name = getattr(
        evidence,
        "metric_name",
        None,
    )

    if metric_name is not None:
        measurements[
            "metric_name"
        ] = metric_name



    # Preserve the calibrated G2 measurement contract.
    if "gap" in measurements:
        measurements["generalization_gap"] = (
            measurements["gap"]
        )

    # Accuracy-specific aliases used by the calibrated
    # underfitting model and explanation builder.
    if metric_name == "accuracy":
        if "train_value" in measurements:
            measurements["training_accuracy"] = (
                measurements["train_value"]
            )

        if "test_value" in measurements:
            measurements["evaluation_accuracy"] = (
                measurements["test_value"]
            )


    dataset_ids: list[str] = []

    for field_name in (
        "train_dataset_id",
        "test_dataset_id",
    ):
        dataset_id = getattr(
            evidence,
            field_name,
            None,
        )

        if dataset_id is not None:
            dataset_ids.append(
                dataset_id
            )

    contexts: list[str] = []

    for field_name in (
        "train_context",
        "test_context",
    ):
        context = getattr(
            evidence,
            field_name,
            None,
        )

        if context is not None:
            contexts.append(
                context
            )

    return DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
        dataset_ids=dataset_ids,
        contexts=contexts,
        measurements=measurements,
        status=_status_to_common(status),
        reason=getattr(
            evidence,
            "reason",
            "",
        ),
        metadata=_common_metadata(
            evidence
        ),
    )


def imbalance_to_evidence(
    evidence: Any,
) -> DiagnosticEvidence:
    """
    Convert ImbalanceEvidence into the common
    DiagnosticEvidence representation.
    """

    status = getattr(
        evidence,
        "status",
        "UNAVAILABLE",
    )

    measurements: dict[str, Any] = {}

    for field_name in (
        "class_count",
        "majority_proportion",
        "minority_proportion",
        "imbalance_ratio",
        "imbalance_ratio_threshold",
    ):
        value = getattr(
            evidence,
            field_name,
            None,
        )

        if value is not None:
            measurements[field_name] = value

    class_distribution = getattr(
        evidence,
        "class_distribution",
        None,
    )

    if class_distribution is not None:
        measurements[
            "class_distribution"
        ] = dict(class_distribution)

    majority_class = getattr(
        evidence,
        "majority_class",
        None,
    )

    if majority_class is not None:
        measurements[
            "majority_class"
        ] = majority_class

    minority_class = getattr(
        evidence,
        "minority_class",
        None,
    )

    if minority_class is not None:
        measurements[
            "minority_class"
        ] = minority_class

    dataset_id = getattr(
        evidence,
        "dataset_id",
        None,
    )

    dataset_ids = (
        [dataset_id]
        if dataset_id is not None
        else []
    )

    context = getattr(
        evidence,
        "context",
        None,
    )

    contexts = (
        [context]
        if context is not None
        else []
    )

    return DiagnosticEvidence(
        evidence_type="imbalance",
        source="target_distribution",
        dataset_ids=dataset_ids,
        contexts=contexts,
        measurements=measurements,
        status=_status_to_common(status),
        reason=getattr(
            evidence,
            "reason",
            "",
        ),
        metadata=_common_metadata(
            evidence
        ),
    )


def distribution_shift_to_evidence(
    evidence: Any,
) -> DiagnosticEvidence:
    """
    Convert DistributionShiftEvidence into the common
    DiagnosticEvidence representation.
    """

    status = getattr(
        evidence,
        "status",
        "UNAVAILABLE",
    )

    measurements: dict[str, Any] = {}

    feature_shifts = getattr(
        evidence,
        "feature_shifts",
        None,
    )

    if feature_shifts is not None:
        measurements[
            "feature_shifts"
        ] = dict(feature_shifts)

    max_shift = getattr(
        evidence,
        "max_shift",
        None,
    )

    if max_shift is not None:
        measurements[
            "max_shift"
        ] = max_shift

    shift_threshold = getattr(
        evidence,
        "shift_threshold",
        None,
    )

    if shift_threshold is not None:
        measurements[
            "shift_threshold"
        ] = shift_threshold

    dataset_ids: list[str] = []

    for field_name in (
        "reference_dataset_id",
        "comparison_dataset_id",
    ):
        dataset_id = getattr(
            evidence,
            field_name,
            None,
        )

        if dataset_id is not None:
            dataset_ids.append(
                dataset_id
            )

    contexts: list[str] = []

    for field_name in (
        "reference_context",
        "comparison_context",
    ):
        context = getattr(
            evidence,
            field_name,
            None,
        )

        if context is not None:
            contexts.append(
                context
            )

    return DiagnosticEvidence(
        evidence_type="distribution_shift",
        source="feature_distribution",
        dataset_ids=dataset_ids,
        contexts=contexts,
        measurements=measurements,
        status=_status_to_common(status),
        reason=getattr(
            evidence,
            "reason",
            "",
        ),
        metadata=_common_metadata(
            evidence
        ),
    )


def leakage_to_evidence(
    evidence: Any,
) -> DiagnosticEvidence:
    """
    Convert LeakageEvidence into the common
    DiagnosticEvidence representation.

    The adapter intentionally preserves leakage-specific
    measurements instead of reducing the finding to a
    simple boolean.
    """

    status = getattr(
        evidence,
        "status",
        "UNAVAILABLE",
    )

    measurements: dict[str, Any] = {}

    # Preserve common scalar diagnostic fields when present.
    for field_name in (
        "score",
        "leakage_score",
        "confidence",
        "overlap",
        "suspicion_score",
        "threshold",
    ):
        value = getattr(
            evidence,
            field_name,
            None,
        )

        if value is not None:
            measurements[field_name] = value

    # Preserve any explicitly available leakage details.
    for field_name in (
        "leakage_type",
        "detected_pattern",
        "source_dataset_id",
        "target_dataset_id",
    ):
        value = getattr(
            evidence,
            field_name,
            None,
        )

        if value is not None:
            measurements[field_name] = value

    dataset_ids: list[str] = []

    # Dataset IDs can differ depending on the leakage
    # evidence implementation, so collect only fields
    # that actually exist.
    for field_name in (
        "dataset_id",
        "source_dataset_id",
        "target_dataset_id",
    ):
        dataset_id = getattr(
            evidence,
            field_name,
            None,
        )

        if dataset_id is not None:
            if dataset_id not in dataset_ids:
                dataset_ids.append(
                    dataset_id
                )

    contexts: list[str] = []

    for field_name in (
        "context",
        "source_context",
        "target_context",
    ):
        context = getattr(
            evidence,
            field_name,
            None,
        )

        if context is not None:
            if context not in contexts:
                contexts.append(
                    context
                )

    return DiagnosticEvidence(
        evidence_type="leakage",
        source="runtime",
        dataset_ids=dataset_ids,
        contexts=contexts,
        measurements=measurements,
        status=_status_to_common(status),
        reason=getattr(
            evidence,
            "reason",
            "",
        ),
        metadata=_common_metadata(
            evidence
        ),
    )




def diagnostic_evidence_to_group(
    evidence: DiagnosticEvidence,
) -> EvidenceGroup:
    """
    Convert common DiagnosticEvidence into the EvidenceGroup
    representation consumed by the Bayesian diagnostic layer.

    This adapter performs only representation normalization.
    It does not determine hypotheses or perform inference.
    """

    if evidence is None:
        raise ValueError(
            "evidence is required"
        )

    try:
        group_name = EVIDENCE_TYPE_TO_GROUP[
            evidence.evidence_type
        ]
    except KeyError as exc:
        raise ValueError(
            "Unsupported diagnostic evidence type: "
            f"{evidence.evidence_type}"
        ) from exc

    if evidence.status == EVIDENCE_OBSERVED:
        status = "observed"

    elif evidence.status == EVIDENCE_INSUFFICIENT:
        status = "unavailable"

    else:
        raise ValueError(
            "Unsupported diagnostic evidence status: "
            f"{evidence.status}"
        )

    metadata = dict(
        evidence.metadata
    )

    measurements = dict(
        evidence.measurements
    )

    if measurements:
        metadata["measurements"] = measurements

    metadata["source"] = evidence.source
    metadata["reason"] = evidence.reason
    metadata["dataset_ids"] = list(
        evidence.dataset_ids
    )
    metadata["contexts"] = list(
        evidence.contexts
    )

    # EvidenceGroup.value is used only when a likelihood model
    # requests the default measurement. Runtime diagnostic
    # evidence is normally measurement-aware, so preserve the
    # complete measurement dictionary separately above.
    value = None

    if "value" in measurements:
        value = measurements["value"]

    return EvidenceGroup(
        name=group_name,
        value=value,
        status=status,
        metadata=metadata,
    )