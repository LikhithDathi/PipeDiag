import pytest

from pipediag.diagnostics.adapters import (
    generalization_to_evidence,
    imbalance_to_evidence,
    distribution_shift_to_evidence,
    leakage_to_evidence,
)

from pipediag.diagnostics.evidence import (
    DiagnosticEvidence,
    EVIDENCE_OBSERVED,
    EVIDENCE_INSUFFICIENT,
)

from pipediag.diagnostics.adapters import (
    diagnostic_evidence_to_group,
)
from pipediag.diagnostics.evidence import (
    DiagnosticEvidence,
    EVIDENCE_OBSERVED,
    EVIDENCE_INSUFFICIENT,
)



from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)
from pipediag.diagnostics.likelihood_registry import (
    LikelihoodModelRegistry,
)

# ======================================================================
# Simple test doubles
# ======================================================================


class GeneralizationResult:

    def __init__(self):
        self.metric_name = "accuracy"
        self.train_value = 0.95
        self.test_value = 0.82
        self.gap = 0.13
        self.metric_direction = "HIGHER_IS_BETTER"

        self.train_dataset_id = "D-TRAIN"
        self.test_dataset_id = "D-TEST"

        self.train_context = "TRAIN"
        self.test_context = "TEST"

        self.status = "COMPUTED"
        self.reason = "TRAIN and TEST evaluations were directly comparable."
        self.metadata = {
            "source": "evaluation"
        }


class ImbalanceResult:

    def __init__(self):
        self.dataset_id = "D-TRAIN"
        self.context = "TRAIN"

        self.class_count = 2

        self.class_distribution = {
            "0": 0.8,
            "1": 0.2,
        }

        self.majority_class = "0"
        self.minority_class = "1"

        self.majority_proportion = 0.8
        self.minority_proportion = 0.2

        self.imbalance_ratio = 4.0
        self.imbalance_ratio_threshold = 3.0

        self.status = "IMBALANCED"
        self.reason = (
            "The observed majority-to-minority class ratio "
            "meets or exceeds the configured imbalance threshold."
        )

        self.metadata = {}


class DistributionShiftResult:

    def __init__(self):
        self.reference_dataset_id = "D-TRAIN"
        self.comparison_dataset_id = "D-TEST"

        self.reference_context = "TRAIN"
        self.comparison_context = "TEST"

        self.feature_shifts = {
            "feature_a": 0.75,
            "feature_b": 0.10,
        }

        self.max_shift = 0.75
        self.shift_threshold = 0.20

        self.status = "SHIFT_DETECTED"
        self.reason = (
            "At least one common feature shows an observed "
            "distribution difference above the configured threshold."
        )

        self.metadata = {}


class LeakageResult:

    def __init__(self):
        self.dataset_id = "D-001"
        self.context = "TRAIN"

        self.leakage_score = 0.85
        self.confidence = 0.90
        self.threshold = 0.70

        self.leakage_type = "target_leakage"
        self.detected_pattern = "target_feature_overlap"

        self.status = "LEAKAGE_DETECTED"
        self.reason = (
            "Potential leakage pattern was observed."
        )

        self.metadata = {
            "method": "runtime_analysis"
        }


class InsufficientResult:

    def __init__(self):
        self.status = "INSUFFICIENT_EVIDENCE"
        self.reason = "Required evidence was unavailable."
        self.metadata = {}


# ======================================================================
# Generalization
# ======================================================================


def test_generalization_is_converted_to_common_evidence():

    result = generalization_to_evidence(
        GeneralizationResult()
    )

    assert isinstance(
        result,
        DiagnosticEvidence,
    )

    assert result.evidence_type == "generalization"
    assert result.source == "evaluation"

    assert result.dataset_ids == [
        "D-TRAIN",
        "D-TEST",
    ]

    assert result.contexts == [
        "TRAIN",
        "TEST",
    ]

    assert result.measurements["train_value"] == 0.95
    assert result.measurements["test_value"] == 0.82
    assert result.measurements["gap"] == 0.13

    # Calibrated G2 measurement aliases.
    assert result.measurements["generalization_gap"] == 0.13
    assert result.measurements["training_accuracy"] == 0.95
    assert result.measurements["evaluation_accuracy"] == 0.82

    assert (
        result.measurements["metric_direction"]
        == "HIGHER_IS_BETTER"
    )

    assert result.measurements["metric_name"] == "accuracy"

    assert result.status == EVIDENCE_OBSERVED


def test_generalization_reason_is_preserved():

    result = generalization_to_evidence(
        GeneralizationResult()
    )

    assert result.reason == (
        "TRAIN and TEST evaluations were directly comparable."
    )


# ======================================================================
# Imbalance
# ======================================================================


def test_imbalance_is_converted_to_common_evidence():

    result = imbalance_to_evidence(
        ImbalanceResult()
    )

    assert isinstance(
        result,
        DiagnosticEvidence,
    )

    assert result.evidence_type == "imbalance"
    assert result.source == "target_distribution"

    assert result.dataset_ids == [
        "D-TRAIN"
    ]

    assert result.contexts == [
        "TRAIN"
    ]

    assert result.measurements["class_count"] == 2
    assert result.measurements["majority_proportion"] == 0.8
    assert result.measurements["minority_proportion"] == 0.2
    assert result.measurements["imbalance_ratio"] == 4.0

    assert result.measurements["majority_class"] == "0"
    assert result.measurements["minority_class"] == "1"

    assert result.measurements["class_distribution"] == {
        "0": 0.8,
        "1": 0.2,
    }

    assert result.status == EVIDENCE_OBSERVED


def test_imbalance_threshold_is_preserved():

    result = imbalance_to_evidence(
        ImbalanceResult()
    )

    assert (
        result.measurements[
            "imbalance_ratio_threshold"
        ]
        == 3.0
    )


# ======================================================================
# Distribution shift
# ======================================================================


def test_distribution_shift_is_converted_to_common_evidence():

    result = distribution_shift_to_evidence(
        DistributionShiftResult()
    )

    assert isinstance(
        result,
        DiagnosticEvidence,
    )

    assert result.evidence_type == "distribution_shift"
    assert result.source == "feature_distribution"

    assert result.dataset_ids == [
        "D-TRAIN",
        "D-TEST",
    ]

    assert result.contexts == [
        "TRAIN",
        "TEST",
    ]

    assert result.measurements["max_shift"] == 0.75
    assert result.measurements["shift_threshold"] == 0.20

    assert result.measurements["feature_shifts"] == {
        "feature_a": 0.75,
        "feature_b": 0.10,
    }

    assert result.status == EVIDENCE_OBSERVED


# ======================================================================
# Leakage
# ======================================================================


def test_leakage_is_converted_to_common_evidence():

    result = leakage_to_evidence(
        LeakageResult()
    )

    assert isinstance(
        result,
        DiagnosticEvidence,
    )

    assert result.evidence_type == "leakage"
    assert result.source == "runtime"

    assert result.dataset_ids == [
        "D-001"
    ]

    assert result.contexts == [
        "TRAIN"
    ]

    assert result.measurements["leakage_score"] == 0.85
    assert result.measurements["confidence"] == 0.90
    assert result.measurements["threshold"] == 0.70

    assert (
        result.measurements["leakage_type"]
        == "target_leakage"
    )

    assert (
        result.measurements["detected_pattern"]
        == "target_feature_overlap"
    )

    assert result.status == EVIDENCE_OBSERVED


# ======================================================================
# Insufficient evidence
# ======================================================================


def test_insufficient_generalization_evidence_is_normalized():

    result = generalization_to_evidence(
        InsufficientResult()
    )

    assert result.status == EVIDENCE_INSUFFICIENT


def test_insufficient_imbalance_evidence_is_normalized():

    result = imbalance_to_evidence(
        InsufficientResult()
    )

    assert result.status == EVIDENCE_INSUFFICIENT


def test_insufficient_distribution_shift_evidence_is_normalized():

    result = distribution_shift_to_evidence(
        InsufficientResult()
    )

    assert result.status == EVIDENCE_INSUFFICIENT


def test_insufficient_leakage_evidence_is_normalized():

    result = leakage_to_evidence(
        InsufficientResult()
    )

    assert result.status == EVIDENCE_INSUFFICIENT


# ======================================================================
# Metadata preservation
# ======================================================================


def test_metadata_is_copied_without_aliasing():

    original = GeneralizationResult()

    result = generalization_to_evidence(
        original
    )

    assert result.metadata == {
        "source": "evaluation"
    }

    assert result.metadata is not original.metadata








def test_diagnostic_evidence_to_group_maps_observed_status():
    evidence = DiagnosticEvidence(
        evidence_type="G3",
        source="distribution",
        status=EVIDENCE_OBSERVED,
    )

    result = diagnostic_evidence_to_group(evidence)

    assert result.name == "G3"
    assert result.status == "observed"


def test_diagnostic_evidence_to_group_maps_insufficient_status():
    evidence = DiagnosticEvidence(
        evidence_type="G3",
        source="distribution",
        status=EVIDENCE_INSUFFICIENT,
        reason="Reference distribution unavailable.",
    )

    result = diagnostic_evidence_to_group(evidence)

    assert result.name == "G3"
    assert result.status == "unavailable"


def test_diagnostic_evidence_to_group_preserves_measurements():
    evidence = DiagnosticEvidence(
        evidence_type="G2",
        source="evaluation",
        measurements={
            "generalization_gap": 0.35,
            "training_accuracy": 0.95,
            "evaluation_accuracy": 0.60,
        },
    )

    result = diagnostic_evidence_to_group(evidence)

    assert result.metadata["measurements"] == {
        "generalization_gap": 0.35,
        "training_accuracy": 0.95,
        "evaluation_accuracy": 0.60,
    }


def test_diagnostic_evidence_to_group_preserves_metadata():
    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
        metadata={
            "metric_name": "accuracy",
            "metric_direction": "HIGHER_IS_BETTER",
        },
    )

    result = diagnostic_evidence_to_group(evidence)

    assert result.name == "G2"
    assert result.metadata["metric_name"] == "accuracy"
    assert (
        result.metadata["metric_direction"]
        == "HIGHER_IS_BETTER"
    )


def test_diagnostic_evidence_to_group_preserves_runtime_context():
    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
        dataset_ids=["D-0001", "D-0002"],
        contexts=["TRAIN", "TEST"],
        reason="TRAIN and TEST evaluations were comparable.",
    )

    result = diagnostic_evidence_to_group(evidence)

    assert result.name == "G2"

    assert result.metadata["dataset_ids"] == [
        "D-0001",
        "D-0002",
    ]

    assert result.metadata["contexts"] == [
        "TRAIN",
        "TEST",
    ]

    assert (
        result.metadata["source"]
        == "evaluation"
    )

    assert (
        result.metadata["reason"]
        == "TRAIN and TEST evaluations were comparable."
    )


def test_diagnostic_evidence_to_group_rejects_unknown_status():
    evidence = DiagnosticEvidence(
        evidence_type="G2",
        source="evaluation",
        status="UNKNOWN_STATUS",
    )

    with pytest.raises(ValueError):
        diagnostic_evidence_to_group(evidence)


def test_diagnostic_evidence_to_group_rejects_none():
    with pytest.raises(ValueError):
        diagnostic_evidence_to_group(None)




def test_diagnostic_evidence_to_group_maps_known_evidence_types():
    cases = {
        "leakage": "G1",
        "generalization": "G2",
        "distribution_shift": "G3",
        "imbalance": "G4",
        "data_quality": "G5",
        "evaluation_problem": "G6",
        "feature_problem": "G7",
        "execution_problem": "G8",
    }

    for evidence_type, expected_group in cases.items():
        evidence = DiagnosticEvidence(
            evidence_type=evidence_type,
            source="test",
        )

        result = diagnostic_evidence_to_group(
            evidence
        )

        assert result.name == expected_group






def test_diagnostic_evidence_to_group_maps_observed_status():
    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
        status=EVIDENCE_OBSERVED,
    )

    result = diagnostic_evidence_to_group(
        evidence
    )

    assert result.name == "G2"
    assert result.status == "observed"






def test_diagnostic_evidence_to_group_maps_insufficient_status():
    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
        status=EVIDENCE_INSUFFICIENT,
        reason="TEST evaluation unavailable.",
    )

    result = diagnostic_evidence_to_group(
        evidence
    )

    assert result.name == "G2"
    assert result.status == "unavailable"
    assert (
        result.metadata["reason"]
        == "TEST evaluation unavailable."
    )







def test_diagnostic_evidence_to_group_preserves_measurements():
    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
        measurements={
            "generalization_gap": 0.35,
            "training_accuracy": 0.95,
            "evaluation_accuracy": 0.60,
        },
    )

    result = diagnostic_evidence_to_group(
        evidence
    )

    assert result.metadata["measurements"] == {
        "generalization_gap": 0.35,
        "training_accuracy": 0.95,
        "evaluation_accuracy": 0.60,
    }








def test_diagnostic_evidence_to_group_preserves_runtime_metadata():
    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
        dataset_ids=["D-001", "D-002"],
        contexts=["TRAIN", "TEST"],
        reason="Comparable evaluations were observed.",
        metadata={
            "metric_name": "accuracy",
        },
    )

    result = diagnostic_evidence_to_group(
        evidence
    )

    assert result.metadata["metric_name"] == "accuracy"

    assert result.metadata["dataset_ids"] == [
        "D-001",
        "D-002",
    ]

    assert result.metadata["contexts"] == [
        "TRAIN",
        "TEST",
    ]

    assert (
        result.metadata["source"]
        == "evaluation"
    )

    assert (
        result.metadata["reason"]
        == "Comparable evaluations were observed."
    )






def test_diagnostic_evidence_to_group_rejects_unknown_status():
    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="test",
        status="UNKNOWN_STATUS",
    )

    with pytest.raises(ValueError):
        diagnostic_evidence_to_group(
            evidence
        )





def test_diagnostic_evidence_to_group_rejects_none():
    with pytest.raises(ValueError):
        diagnostic_evidence_to_group(
            None
        )






def test_diagnostic_evidence_to_group_integrates_with_likelihood_evaluator():
    registry = LikelihoodModelRegistry()

    registry.register(
        "G2",
        "H2",
        GaussianLikelihoodModel(
            mean_true=0.40,
            std_true=0.10,
            mean_false=0.05,
            std_false=0.10,
            measurement="generalization_gap",
        ),
    )

    registry.register(
        "G2",
        "H7",
        GaussianLikelihoodModel(
            mean_true=0.10,
            std_true=0.10,
            mean_false=0.40,
            std_false=0.10,
            measurement="generalization_gap",
        ),
    )

    evidence = DiagnosticEvidence(
        evidence_type="generalization",
        source="evaluation",
        measurements={
            "generalization_gap": 0.35,
        },
    )

    group = diagnostic_evidence_to_group(
        evidence
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=(
            EVIDENCE_HYPOTHESIS_MAP
        ),
    )

    log_ratios = evaluator.evaluate_log_ratios(
        group
    )

    assert set(log_ratios) == {
        "H2",
        "H7",
    }

    assert log_ratios["H2"] > 0
    assert log_ratios["H7"] < 0