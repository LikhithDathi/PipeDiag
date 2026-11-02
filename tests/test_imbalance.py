from pipediag.diagnostics.imbalance import (
    ImbalanceDiagnostic,
    IMBALANCED,
    NO_IMBALANCE,
    INSUFFICIENT_EVIDENCE,
    NOT_APPLICABLE,
)
from pipediag.evidence.store import (
    TRAIN,
    TEST,
    UNKNOWN,
)


def make_target_evidence(
    *,
    dataset_id="D-001",
    context=TRAIN,
    class_distribution=None,
):
    if class_distribution is None:
        class_distribution = {
            "0": 0.5,
            "1": 0.5,
        }

    return {
        "dataset_id": dataset_id,
        "context": context,
        "class_distribution": class_distribution,
    }


# ======================================================================
# Basic evidence handling
# ======================================================================


def test_missing_target_evidence_is_insufficient():

    diagnostic = ImbalanceDiagnostic()

    result = diagnostic.analyze(
        target_evidence=None,
    )

    assert result.status == INSUFFICIENT_EVIDENCE


def test_empty_target_evidence_is_insufficient():

    diagnostic = ImbalanceDiagnostic()

    result = diagnostic.analyze(
        target_evidence={},
    )

    assert result.status == INSUFFICIENT_EVIDENCE


def test_unknown_context_does_not_prevent_distribution_analysis():

    diagnostic = ImbalanceDiagnostic()

    evidence = make_target_evidence(
        context=UNKNOWN,
        class_distribution={
            "0": 0.8,
            "1": 0.2,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.status == IMBALANCED
    assert result.context == UNKNOWN


# ======================================================================
# Binary classification
# ======================================================================


def test_balanced_binary_distribution():

    diagnostic = ImbalanceDiagnostic(
        imbalance_ratio_threshold=3.0,
    )

    evidence = make_target_evidence(
        class_distribution={
            "0": 0.5,
            "1": 0.5,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.status == NO_IMBALANCE
    assert result.class_count == 2
    assert result.majority_class == "0"
    assert result.minority_class == "0"
    assert result.majority_proportion == 0.5
    assert result.minority_proportion == 0.5
    assert result.imbalance_ratio == 1.0


def test_imbalanced_binary_distribution():

    diagnostic = ImbalanceDiagnostic(
        imbalance_ratio_threshold=3.0,
    )

    evidence = make_target_evidence(
        class_distribution={
            "0": 0.8,
            "1": 0.2,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.status == IMBALANCED
    assert result.class_count == 2
    assert result.majority_class == "0"
    assert result.minority_class == "1"
    assert result.majority_proportion == 0.8
    assert result.minority_proportion == 0.2
    assert result.imbalance_ratio == 4.0


# ======================================================================
# Multiclass classification
# ======================================================================


def test_multiclass_distribution_is_supported():

    diagnostic = ImbalanceDiagnostic(
        imbalance_ratio_threshold=3.0,
    )

    evidence = make_target_evidence(
        class_distribution={
            "A": 0.6,
            "B": 0.25,
            "C": 0.15,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.status == IMBALANCED
    assert result.class_count == 3
    assert result.majority_class == "A"
    assert result.minority_class == "C"
    assert result.majority_proportion == 0.6
    assert result.minority_proportion == 0.15
    assert result.imbalance_ratio == 4.0


def test_complete_distribution_is_preserved():

    diagnostic = ImbalanceDiagnostic()

    distribution = {
        "A": 0.55,
        "B": 0.30,
        "C": 0.15,
    }

    evidence = make_target_evidence(
        class_distribution=distribution,
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.class_distribution == distribution


# ======================================================================
# Threshold behaviour
# ======================================================================


def test_ratio_below_threshold_is_not_imbalanced():

    diagnostic = ImbalanceDiagnostic(
        imbalance_ratio_threshold=3.0,
    )

    evidence = make_target_evidence(
        class_distribution={
            "0": 0.7,
            "1": 0.3,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.imbalance_ratio == 7 / 3
    assert result.status == NO_IMBALANCE


def test_ratio_above_threshold_is_imbalanced():

    diagnostic = ImbalanceDiagnostic(
        imbalance_ratio_threshold=3.0,
    )

    evidence = make_target_evidence(
        class_distribution={
            "0": 0.75,
            "1": 0.25,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.imbalance_ratio == 3.0
    assert result.status == IMBALANCED


# ======================================================================
# Invalid distributions
# ======================================================================


def test_missing_distribution_is_insufficient():

    diagnostic = ImbalanceDiagnostic()

    evidence = {
        "dataset_id": "D-001",
        "context": TRAIN,
    }

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.status == INSUFFICIENT_EVIDENCE


def test_empty_distribution_is_insufficient():

    diagnostic = ImbalanceDiagnostic()

    evidence = make_target_evidence(
        class_distribution={},
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.status == INSUFFICIENT_EVIDENCE


def test_single_class_is_not_class_imbalance():

    diagnostic = ImbalanceDiagnostic()

    evidence = make_target_evidence(
        class_distribution={
            "0": 1.0,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.status == NOT_APPLICABLE


# ======================================================================
# Evidence preservation
# ======================================================================


def test_dataset_id_is_preserved():

    diagnostic = ImbalanceDiagnostic()

    evidence = make_target_evidence(
        dataset_id="D-123",
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.dataset_id == "D-123"


def test_context_is_preserved():

    diagnostic = ImbalanceDiagnostic()

    evidence = make_target_evidence(
        context=TEST,
        class_distribution={
            "0": 0.6,
            "1": 0.4,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.context == TEST


# ======================================================================
# No causal claims
# ======================================================================


def test_imbalance_diagnostic_does_not_claim_model_failure():

    diagnostic = ImbalanceDiagnostic()

    evidence = make_target_evidence(
        class_distribution={
            "0": 0.9,
            "1": 0.1,
        },
    )

    result = diagnostic.analyze(
        target_evidence=evidence,
    )

    assert result.status == IMBALANCED

    assert "performance" not in result.reason.lower()
    assert "model failure" not in result.reason.lower()
    assert "overfitting" not in result.reason.lower()