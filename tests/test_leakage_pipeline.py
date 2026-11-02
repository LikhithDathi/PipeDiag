import numpy as np
import pytest

from experiments.leakage_evidence import leakage_score
from experiments.leakage_pipeline import (
    LeakagePipelineResult,
    run_leakage_pipeline,
)


def create_result():
    return run_leakage_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )


def test_leakage_pipeline_returns_result():
    result = create_result()

    assert isinstance(
        result,
        LeakagePipelineResult,
    )


def test_leakage_feature_is_added():
    result = create_result()

    assert result.X_train.shape[1] == 6
    assert result.X_test.shape[1] == 6


def test_leakage_feature_is_last_feature():
    result = create_result()

    assert (
        result.leakage_feature_index
        == result.X_train.shape[1] - 1
    )


def test_leakage_feature_contains_training_target():
    result = create_result()

    leakage_feature = (
        result.X_train[
            :,
            result.leakage_feature_index,
        ]
    )

    assert np.array_equal(
        leakage_feature,
        result.y_train,
    )


def test_leakage_feature_contains_test_target():
    result = create_result()

    leakage_feature = (
        result.X_test[
            :,
            result.leakage_feature_index,
        ]
    )

    assert np.array_equal(
        leakage_feature,
        result.y_test,
    )


def test_predictions_have_correct_shape():
    result = create_result()

    assert result.predictions.shape == (
        result.y_test.shape
    )


def test_probabilities_have_correct_shape():
    result = create_result()

    assert result.probabilities.shape == (
        len(result.y_test),
        2,
    )


def test_accuracy_is_valid():
    result = create_result()

    assert 0.0 <= result.accuracy <= 1.0


def test_leakage_pipeline_is_reproducible():
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

    assert np.array_equal(
        first.X_train,
        second.X_train,
    )

    assert np.array_equal(
        first.X_test,
        second.X_test,
    )

    assert np.array_equal(
        first.predictions,
        second.predictions,
    )


def test_leakage_pipeline_stores_random_state():
    result = create_result()

    assert result.random_state == 42


def test_leakage_pipeline_rejects_invalid_sample_count():
    with pytest.raises(ValueError):
        run_leakage_pipeline(
            n_samples=0,
            n_features=5,
        )


def test_leakage_pipeline_rejects_invalid_feature_count():
    with pytest.raises(ValueError):
        run_leakage_pipeline(
            n_samples=200,
            n_features=0,
        )


def test_leakage_pipeline_rejects_invalid_test_size():
    with pytest.raises(ValueError):
        run_leakage_pipeline(
            n_samples=200,
            n_features=5,
            test_size=1.0,
        )





def test_default_leakage_strength_preserves_exact_target():
    result = run_leakage_pipeline(
        n_samples=300,
        n_features=8,
        random_state=42,
    )

    leakage_column = result.X_train[
        :, result.leakage_feature_index
    ]

    assert np.array_equal(
        leakage_column,
        result.y_train,
    )


def test_reduced_leakage_strength_produces_non_perfect_leakage():
    result = run_leakage_pipeline(
        n_samples=300,
        n_features=8,
        random_state=42,
        leakage_strength=0.85,
    )

    leakage_column = result.X_train[
        :, result.leakage_feature_index
    ]

    assert not np.array_equal(
        leakage_column,
        result.y_train,
    )


def test_invalid_leakage_strength_is_rejected():
    with pytest.raises(ValueError):
        run_leakage_pipeline(
            n_samples=300,
            n_features=8,
            random_state=42,
            leakage_strength=0.0,
        )


def test_leakage_strength_above_one_is_rejected():
    with pytest.raises(ValueError):
        run_leakage_pipeline(
            n_samples=300,
            n_features=8,
            random_state=42,
            leakage_strength=1.1,
        )




def test_leakage_feature_strength_increases_correlation():
    strengths = (0.01, 0.25, 0.50, 0.75, 1.0)

    scores = []

    for strength in strengths:
        result = run_leakage_pipeline(
            n_samples=500,
            n_features=10,
            leakage_strength=strength,
            random_state=42,
        )

        score = leakage_score(
            result.X_train,
            result.y_train,
            leakage_feature_index=result.leakage_feature_index,
        )

        scores.append(score)

    assert all(
        earlier <= later
        for earlier, later in zip(scores, scores[1:])
    )