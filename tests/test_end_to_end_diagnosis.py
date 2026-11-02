import numpy as np

from pipediag.acquisition.evaluation import EvaluationEvidence
from pipediag.diagnostics.adapters import (
    diagnostic_evidence_to_group,
    generalization_to_evidence,
)
from pipediag.diagnostics.generalization import (
    HIGHER_IS_BETTER,
    GeneralizationDiagnostic,
)


def make_evaluations():
    return [
        EvaluationEvidence(
            metric_name="accuracy",
            metric_value=0.96,
            evaluation_context="TRAIN",
            feature_dataset_id="D-TRAIN-X",
            target_dataset_id="D-TRAIN-Y",
            sample_count=100,
            source="explicit_metric",
        ),
        EvaluationEvidence(
            metric_name="accuracy",
            metric_value=0.61,
            evaluation_context="TEST",
            feature_dataset_id="D-TEST-X",
            target_dataset_id="D-TEST-Y",
            sample_count=100,
            source="explicit_metric",
        ),
    ]


def test_evaluation_evidence_produces_generalization_diagnostic_evidence():
    evaluations = make_evaluations()

    diagnostic = GeneralizationDiagnostic()

    result = diagnostic.analyze(
        evaluations,
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == "COMPUTED"
    assert result.metric_name == "accuracy"
    assert result.train_value == 0.96
    assert result.test_value == 0.61
    assert np.isclose(result.gap, 0.35)


def test_generalization_diagnostic_converts_to_g2():
    evaluations = make_evaluations()

    diagnostic = GeneralizationDiagnostic()

    generalization = diagnostic.analyze(
        evaluations,
        metric_direction=HIGHER_IS_BETTER,
    )

    evidence = generalization_to_evidence(generalization)

    assert evidence.evidence_type == "generalization"
    assert evidence.status == "OBSERVED"
    assert evidence.measurements["gap"] == 0.35

    group = diagnostic_evidence_to_group(evidence)

    assert group.name == "G2"
    assert group.status == "observed"
    assert group.metadata["measurements"]["gap"] == 0.35