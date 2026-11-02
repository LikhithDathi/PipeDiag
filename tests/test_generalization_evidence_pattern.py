import pytest

from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import (
    run_overfitting_pipeline,
)


def test_g2_contains_training_accuracy():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(result)

    assert "training_accuracy" in evidence.metadata


def test_g2_contains_evaluation_accuracy():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(result)

    assert "evaluation_accuracy" in evidence.metadata


def test_g2_training_accuracy_matches_result():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(result)

    assert evidence.metadata["training_accuracy"] == pytest.approx(
        result.train_accuracy
    )


def test_g2_evaluation_accuracy_matches_result():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(result)

    assert evidence.metadata["evaluation_accuracy"] == pytest.approx(
        result.test_accuracy
    )


def test_g2_gap_matches_training_minus_evaluation():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(result)

    expected_gap = (
        result.train_accuracy - result.test_accuracy
    )

    assert evidence.value == pytest.approx(expected_gap)


def test_overfitting_pattern_has_high_training_accuracy():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(result)

    assert evidence.metadata["training_accuracy"] > 0.8


def test_underfitting_configuration_has_lower_training_accuracy():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=1,
    )

    evidence = collect_generalization_evidence(result)

    assert evidence.metadata["training_accuracy"] < 0.9


def test_evidence_is_observed():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(result)

    assert evidence.status == "observed"