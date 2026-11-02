import numpy as np
import pytest

from experiments.clean_pipeline import (
    CleanPipelineResult,
    run_clean_pipeline,
)


def test_clean_pipeline_returns_result():
    result = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    assert isinstance(
        result,
        CleanPipelineResult,
    )


def test_clean_pipeline_has_train_and_test_data():
    result = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    assert result.X_train.shape[0] > 0
    assert result.X_test.shape[0] > 0
    assert result.y_train.shape[0] == result.X_train.shape[0]
    assert result.y_test.shape[0] == result.X_test.shape[0]


def test_clean_pipeline_preserves_feature_count():
    result = run_clean_pipeline(
        n_samples=200,
        n_features=7,
        random_state=42,
    )

    assert result.X_train.shape[1] == 7
    assert result.X_test.shape[1] == 7


def test_clean_pipeline_generates_predictions():
    result = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    assert result.predictions.shape == result.y_test.shape


def test_clean_pipeline_generates_probabilities():
    result = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    assert result.probabilities.shape == (
        len(result.y_test),
        2,
    )


def test_clean_pipeline_accuracy_is_valid():
    result = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    assert 0.0 <= result.accuracy <= 1.0


def test_clean_pipeline_is_reproducible():
    first = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    second = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    assert np.array_equal(
        first.X_train,
        second.X_train,
    )

    assert np.array_equal(
        first.y_train,
        second.y_train,
    )

    assert np.array_equal(
        first.predictions,
        second.predictions,
    )

    assert first.accuracy == pytest.approx(
        second.accuracy
    )


def test_clean_pipeline_changes_with_random_state():
    first = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    second = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=99,
    )

    assert not np.array_equal(
        first.X_train,
        second.X_train,
    )


def test_clean_pipeline_stores_random_state():
    result = run_clean_pipeline(
        n_samples=200,
        n_features=5,
        random_state=42,
    )

    assert result.random_state == 42


def test_clean_pipeline_rejects_invalid_sample_count():
    with pytest.raises(ValueError):
        run_clean_pipeline(
            n_samples=0,
            n_features=5,
            random_state=42,
        )


def test_clean_pipeline_rejects_invalid_feature_count():
    with pytest.raises(ValueError):
        run_clean_pipeline(
            n_samples=200,
            n_features=0,
            random_state=42,
        )


def test_clean_pipeline_rejects_invalid_test_size():
    with pytest.raises(ValueError):
        run_clean_pipeline(
            n_samples=200,
            n_features=5,
            test_size=1.0,
            random_state=42,
        )