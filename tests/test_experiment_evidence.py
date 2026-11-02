import numpy as np
import pytest

from experiments.clean_pipeline import run_clean_pipeline
from experiments.evidence import (
    collect_clean_pipeline_evidence,
)
from pipediag.diagnostics.evidence_group import EvidenceGroup


def create_pipeline_result():
    return run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )


def test_collects_evidence_groups():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    assert evidence


def test_returns_evidence_groups():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    assert all(
        isinstance(item, EvidenceGroup)
        for item in evidence
    )


def test_contains_performance_evidence():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    names = {
        item.name
        for item in evidence
    }

    assert "G2" in names


def test_contains_class_distribution_evidence():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    names = {
        item.name
        for item in evidence
    }

    assert "G4" in names


def test_contains_data_quality_evidence():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    names = {
        item.name
        for item in evidence
    }

    assert "G5" in names


def test_contains_feature_evidence():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    names = {
        item.name
        for item in evidence
    }

    assert "G7" in names


def test_all_available_evidence_is_observed():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    for item in evidence:
        assert item.status == "observed"


def test_performance_evidence_is_accuracy():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    performance = next(
        item
        for item in evidence
        if item.name == "G2"
    )

    assert performance.value == pytest.approx(
        result.accuracy
    )


def test_class_distribution_evidence_is_imbalance_ratio():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    class_evidence = next(
        item
        for item in evidence
        if item.name == "G4"
    )

    counts = np.bincount(
        result.y_train
    )

    expected = (
        min(counts)
        / max(counts)
    )

    assert class_evidence.value == pytest.approx(
        expected
    )


def test_data_quality_evidence_is_missing_value_rate():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    quality = next(
        item
        for item in evidence
        if item.name == "G5"
    )

    expected = float(
        np.isnan(result.X_train).mean()
    )

    assert quality.value == pytest.approx(
        expected
    )


def test_feature_evidence_is_feature_count():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    feature = next(
        item
        for item in evidence
        if item.name == "G7"
    )

    assert feature.value == result.X_train.shape[1]


def test_clean_pipeline_has_no_missing_values():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    quality = next(
        item
        for item in evidence
        if item.name == "G5"
    )

    assert quality.value == 0.0


def test_clean_pipeline_has_balanced_classes():
    result = create_pipeline_result()

    evidence = collect_clean_pipeline_evidence(
        result
    )

    class_evidence = next(
        item
        for item in evidence
        if item.name == "G4"
    )

    assert 0.0 < class_evidence.value <= 1.0