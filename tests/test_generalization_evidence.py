import pytest

from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import (
    run_overfitting_pipeline,
)


def _result():
    return run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )


def test_generalization_evidence_returns_evidence_group():
    result = _result()

    evidence = collect_generalization_evidence(
        result
    )

    assert evidence.name == "G2"
    assert evidence.status == "observed"


def test_generalization_gap_is_used_as_evidence_value():
    result = _result()

    evidence = collect_generalization_evidence(
        result
    )

    assert evidence.value == pytest.approx(
        result.generalization_gap
    )


def test_training_accuracy_is_preserved_in_metadata():
    result = _result()

    evidence = collect_generalization_evidence(
        result
    )

    assert evidence.metadata[
        "training_accuracy"
    ] == pytest.approx(
        result.train_accuracy
    )


def test_evaluation_accuracy_is_preserved_in_metadata():
    result = _result()

    evidence = collect_generalization_evidence(
        result
    )

    assert evidence.metadata[
        "evaluation_accuracy"
    ] == pytest.approx(
        result.test_accuracy
    )


def test_metric_is_recorded():
    result = _result()

    evidence = collect_generalization_evidence(
        result
    )

    assert evidence.metadata["metric"] == (
        "generalization_gap"
    )


def test_source_is_recorded():
    result = _result()

    evidence = collect_generalization_evidence(
        result
    )

    assert evidence.metadata["source"] == (
        "overfitting_pipeline"
    )


def test_none_result_is_rejected():
    with pytest.raises(ValueError):
        collect_generalization_evidence(None)


def test_evidence_value_is_finite():
    result = _result()

    evidence = collect_generalization_evidence(
        result
    )

    assert evidence.value == pytest.approx(
        float(evidence.value)
    )