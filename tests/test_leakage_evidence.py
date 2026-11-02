import numpy as np
import pytest

from experiments.leakage_evidence import (
    leakage_score,
    collect_leakage_evidence,
)
from experiments.leakage_pipeline import (
    run_leakage_pipeline,
)
from experiments.clean_pipeline import (
    run_clean_pipeline,
)
from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
)


def test_leakage_score_returns_float():
    result = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    score = leakage_score(
        result.X_train,
        result.y_train,
    )

    assert isinstance(score, float)


def test_leakage_score_is_between_zero_and_one():
    result = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    score = leakage_score(
        result.X_train,
        result.y_train,
    )

    assert 0.0 <= score <= 1.0


def test_injected_leakage_produces_near_perfect_score():
    result = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    score = leakage_score(
        result.X_train,
        result.y_train,
    )

    assert score == pytest.approx(
        1.0,
        abs=1e-10,
    )


def test_clean_pipeline_has_lower_leakage_score():
    clean = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    leakage = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    clean_score = leakage_score(
        clean.X_train,
        clean.y_train,
    )

    leakage_score_value = leakage_score(
        leakage.X_train,
        leakage.y_train,
    )

    assert leakage_score_value > clean_score


def test_collect_leakage_evidence_returns_evidence_group():
    result = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    evidence = collect_leakage_evidence(
        result
    )

    assert isinstance(
        evidence,
        EvidenceGroup,
    )


def test_collect_leakage_evidence_uses_g1():
    result = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    evidence = collect_leakage_evidence(
        result
    )

    assert evidence.name == "G1"


def test_collect_leakage_evidence_is_observed():
    result = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    evidence = collect_leakage_evidence(
        result
    )

    assert evidence.status == "observed"


def test_collect_leakage_evidence_matches_score():
    result = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    evidence = collect_leakage_evidence(
        result
    )

    expected = leakage_score(
        result.X_train,
        result.y_train,
    )

    assert evidence.value == pytest.approx(
        expected
    )


def test_leakage_score_rejects_empty_features():
    with pytest.raises(ValueError):
        leakage_score(
            np.empty((0, 5)),
            np.array([]),
        )


def test_leakage_score_rejects_mismatched_samples():
    with pytest.raises(ValueError):
        leakage_score(
            np.zeros((10, 5)),
            np.zeros(9),
        )


def test_leakage_score_rejects_non_2d_features():
    with pytest.raises(ValueError):
        leakage_score(
            np.zeros(10),
            np.zeros(10),
        )


def test_leakage_score_rejects_non_1d_target():
    with pytest.raises(ValueError):
        leakage_score(
            np.zeros((10, 5)),
            np.zeros((10, 1)),
        )


def test_leakage_score_is_reproducible():
    first = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    second = run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    first_score = leakage_score(
        first.X_train,
        first.y_train,
    )

    second_score = leakage_score(
        second.X_train,
        second.y_train,
    )

    assert first_score == pytest.approx(
        second_score
    )