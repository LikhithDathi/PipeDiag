from pipediag.diagnostics.distribution_shift import (
    DistributionShiftDiagnostic,
    SHIFT_DETECTED,
    NO_SHIFT,
    INSUFFICIENT_EVIDENCE,
    NOT_APPLICABLE,
)
from pipediag.evidence.store import (
    TRAIN,
    TEST,
    UNKNOWN,
)


def make_distribution_evidence(
    *,
    dataset_id="D-001",
    context=TRAIN,
    distributions=None,
):
    if distributions is None:
        distributions = {
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [10.0, 11.0, 12.0, 13.0],
        }

    return {
        "dataset_id": dataset_id,
        "context": context,
        "distributions": distributions,
    }


# ======================================================================
# Basic evidence handling
# ======================================================================


def test_missing_reference_evidence_is_insufficient():

    diagnostic = DistributionShiftDiagnostic()

    result = diagnostic.analyze(
        reference_evidence=None,
        comparison_evidence=make_distribution_evidence(
            context=TEST,
        ),
    )

    assert result.status == INSUFFICIENT_EVIDENCE


def test_missing_comparison_evidence_is_insufficient():

    diagnostic = DistributionShiftDiagnostic()

    result = diagnostic.analyze(
        reference_evidence=make_distribution_evidence(
            context=TRAIN,
        ),
        comparison_evidence=None,
    )

    assert result.status == INSUFFICIENT_EVIDENCE


def test_empty_reference_evidence_is_insufficient():

    diagnostic = DistributionShiftDiagnostic()

    result = diagnostic.analyze(
        reference_evidence={},
        comparison_evidence=make_distribution_evidence(
            context=TEST,
        ),
    )

    assert result.status == INSUFFICIENT_EVIDENCE


def test_empty_comparison_evidence_is_insufficient():

    diagnostic = DistributionShiftDiagnostic()

    result = diagnostic.analyze(
        reference_evidence=make_distribution_evidence(
            context=TRAIN,
        ),
        comparison_evidence={},
    )

    assert result.status == INSUFFICIENT_EVIDENCE


# ======================================================================
# Context handling
# ======================================================================


def test_contexts_are_preserved():

    diagnostic = DistributionShiftDiagnostic()

    reference = make_distribution_evidence(
        dataset_id="D-TRAIN",
        context=TRAIN,
    )

    comparison = make_distribution_evidence(
        dataset_id="D-TEST",
        context=TEST,
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.reference_context == TRAIN
    assert result.comparison_context == TEST


def test_unknown_context_does_not_prevent_distribution_analysis():

    diagnostic = DistributionShiftDiagnostic()

    reference = make_distribution_evidence(
        context=UNKNOWN,
    )

    comparison = make_distribution_evidence(
        context=UNKNOWN,
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.status == NO_SHIFT


# ======================================================================
# No shift
# ======================================================================


def test_identical_distributions_have_no_shift():

    diagnostic = DistributionShiftDiagnostic(
        shift_threshold=0.1,
    )

    reference = make_distribution_evidence(
        distributions={
            "feature_1": [1.0, 2.0, 3.0, 4.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_1": [1.0, 2.0, 3.0, 4.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.status == NO_SHIFT
    assert result.max_shift == 0.0


def test_small_shift_below_threshold_is_not_detected():

    diagnostic = DistributionShiftDiagnostic(
        shift_threshold=0.2,
    )

    reference = make_distribution_evidence(
        distributions={
            "feature_1": [0.0, 0.0, 1.0, 1.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_1": [0.0, 0.1, 1.0, 1.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.status == NO_SHIFT


# ======================================================================
# Shift detection
# ======================================================================


def test_large_shift_is_detected():

    diagnostic = DistributionShiftDiagnostic(
        shift_threshold=0.2,
    )

    reference = make_distribution_evidence(
        distributions={
            "feature_1": [0.0, 0.0, 0.0, 1.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_1": [1.0, 1.0, 1.0, 1.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.status == SHIFT_DETECTED
    assert result.max_shift > 0.2


def test_shift_is_assessed_per_feature():

    diagnostic = DistributionShiftDiagnostic(
        shift_threshold=0.2,
    )

    reference = make_distribution_evidence(
        distributions={
            "stable": [1.0, 1.0, 1.0, 1.0],
            "shifted": [0.0, 0.0, 0.0, 1.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "stable": [1.0, 1.0, 1.0, 1.0],
            "shifted": [1.0, 1.0, 1.0, 1.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.status == SHIFT_DETECTED
    assert "shifted" in result.feature_shifts
    assert "stable" in result.feature_shifts


# ======================================================================
# Feature matching
# ======================================================================


def test_features_missing_from_comparison_are_not_silently_compared():

    diagnostic = DistributionShiftDiagnostic()

    reference = make_distribution_evidence(
        distributions={
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_a": [1.0, 2.0, 3.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert "feature_a" in result.feature_shifts
    assert "feature_b" not in result.feature_shifts


def test_no_common_features_is_insufficient():

    diagnostic = DistributionShiftDiagnostic()

    reference = make_distribution_evidence(
        distributions={
            "feature_a": [1.0, 2.0, 3.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_b": [1.0, 2.0, 3.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.status == INSUFFICIENT_EVIDENCE


# ======================================================================
# Dataset identity
# ======================================================================


def test_dataset_ids_are_preserved():

    diagnostic = DistributionShiftDiagnostic()

    reference = make_distribution_evidence(
        dataset_id="D-TRAIN",
    )

    comparison = make_distribution_evidence(
        dataset_id="D-TEST",
        context=TEST,
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.reference_dataset_id == "D-TRAIN"
    assert result.comparison_dataset_id == "D-TEST"


# ======================================================================
# Evidence preservation
# ======================================================================


def test_feature_shift_values_are_preserved():

    diagnostic = DistributionShiftDiagnostic(
        shift_threshold=0.2,
    )

    reference = make_distribution_evidence(
        distributions={
            "feature_1": [0.0, 0.0, 0.0, 1.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_1": [1.0, 1.0, 1.0, 1.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert "feature_1" in result.feature_shifts
    assert result.feature_shifts["feature_1"] == result.max_shift


def test_max_shift_is_the_largest_feature_shift():

    diagnostic = DistributionShiftDiagnostic(
        shift_threshold=0.2,
    )

    reference = make_distribution_evidence(
        distributions={
            "feature_a": [0.0, 0.0, 1.0, 1.0],
            "feature_b": [0.0, 0.0, 0.0, 1.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_a": [0.0, 0.5, 0.5, 1.0],
            "feature_b": [1.0, 1.0, 1.0, 1.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.max_shift == max(
        result.feature_shifts.values()
    )


# ======================================================================
# Invalid evidence
# ======================================================================


def test_invalid_distribution_is_insufficient():

    diagnostic = DistributionShiftDiagnostic()

    reference = make_distribution_evidence(
        distributions={
            "feature_1": [1.0, 2.0, "invalid"],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_1": [1.0, 2.0, 3.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.status == INSUFFICIENT_EVIDENCE


# ======================================================================
# No causal claims
# ======================================================================


def test_distribution_shift_does_not_claim_model_failure():

    diagnostic = DistributionShiftDiagnostic()

    reference = make_distribution_evidence(
        distributions={
            "feature_1": [0.0, 0.0, 0.0, 1.0],
        },
    )

    comparison = make_distribution_evidence(
        context=TEST,
        distributions={
            "feature_1": [1.0, 1.0, 1.0, 1.0],
        },
    )

    result = diagnostic.analyze(
        reference_evidence=reference,
        comparison_evidence=comparison,
    )

    assert result.status == SHIFT_DETECTED

    assert "performance" not in result.reason.lower()
    assert "model failure" not in result.reason.lower()
    assert "overfitting" not in result.reason.lower()