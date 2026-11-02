from pipediag.diagnostics.evidence import (
    DiagnosticEvidence,
    EVIDENCE_OBSERVED,
    EVIDENCE_INSUFFICIENT,
)


def test_diagnostic_evidence_stores_basic_information():

    evidence = DiagnosticEvidence(
        evidence_type="imbalance",
        source="target_distribution",
        dataset_ids=["D-001"],
        contexts=["TRAIN"],
        measurements={
            "imbalance_ratio": 4.0,
        },
        status=EVIDENCE_OBSERVED,
        reason="Observed target distribution is imbalanced.",
    )

    assert evidence.evidence_type == "imbalance"
    assert evidence.source == "target_distribution"
    assert evidence.dataset_ids == ["D-001"]
    assert evidence.contexts == ["TRAIN"]
    assert evidence.measurements["imbalance_ratio"] == 4.0
    assert evidence.status == EVIDENCE_OBSERVED


def test_dataset_ids_default_to_empty():

    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
    )

    assert evidence.dataset_ids == []


def test_contexts_default_to_empty():

    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
    )

    assert evidence.contexts == []


def test_measurements_default_to_empty():

    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
    )

    assert evidence.measurements == {}


def test_metadata_defaults_to_empty():

    evidence = DiagnosticEvidence(
        evidence_type="leakage",
        source="runtime",
    )

    assert evidence.metadata == {}


def test_insufficient_evidence_can_be_represented():

    evidence = DiagnosticEvidence(
        evidence_type="distribution_shift",
        source="feature_distribution",
        status=EVIDENCE_INSUFFICIENT,
        reason="Comparison dataset was not available.",
    )

    assert evidence.status == EVIDENCE_INSUFFICIENT
    assert evidence.reason == (
        "Comparison dataset was not available."
    )


def test_measurements_preserve_multiple_values():

    evidence = DiagnosticEvidence(
        evidence_type="distribution_shift",
        source="feature_distribution",
        measurements={
            "max_shift": 0.75,
            "feature_a": 0.75,
            "feature_b": 0.12,
        },
    )

    assert evidence.measurements["max_shift"] == 0.75
    assert evidence.measurements["feature_a"] == 0.75
    assert evidence.measurements["feature_b"] == 0.12


def test_metadata_can_preserve_additional_context():

    evidence = DiagnosticEvidence(
        evidence_type="imbalance",
        source="target_distribution",
        metadata={
            "threshold": 3.0,
            "class_count": 3,
        },
    )

    assert evidence.metadata["threshold"] == 3.0
    assert evidence.metadata["class_count"] == 3