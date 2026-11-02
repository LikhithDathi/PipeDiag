import pytest

from experiments.overfitting_pipeline import (
    OverfittingPipelineResult,
    run_overfitting_pipeline,
)


def test_overfitting_pipeline_returns_result():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert isinstance(
        result,
        OverfittingPipelineResult,
    )


def test_training_and_test_shapes_are_consistent():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert result.X_train.shape[0] == result.y_train.shape[0]
    assert result.X_test.shape[0] == result.y_test.shape[0]

    assert result.X_train.shape[1] == 10
    assert result.X_test.shape[1] == 10


def test_accuracy_values_are_valid():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert 0.0 <= result.train_accuracy <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0


def test_generalization_gap_matches_accuracies():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    expected_gap = (
        result.train_accuracy
        - result.test_accuracy
    )

    assert result.generalization_gap == pytest.approx(
        expected_gap
    )


def test_unrestricted_tree_can_fit_training_data_well():
    result = run_overfitting_pipeline(
        n_samples=200,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    assert result.train_accuracy >= 0.95


def test_restricted_tree_has_smaller_capacity():
    unrestricted = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    restricted = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=2,
    )

    assert (
        restricted.train_accuracy
        <= unrestricted.train_accuracy
    )


def test_max_depth_is_preserved():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=2,
    )

    assert result.max_depth == 2


def test_random_state_is_preserved():
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=123,
    )

    assert result.random_state == 123


def test_invalid_sample_count_is_rejected():
    with pytest.raises(ValueError):
        run_overfitting_pipeline(
            n_samples=0,
            n_features=10,
        )


def test_invalid_feature_count_is_rejected():
    with pytest.raises(ValueError):
        run_overfitting_pipeline(
            n_samples=500,
            n_features=0,
        )


def test_invalid_test_size_is_rejected():
    with pytest.raises(ValueError):
        run_overfitting_pipeline(
            n_samples=500,
            n_features=10,
            test_size=0.0,
        )

    with pytest.raises(ValueError):
        run_overfitting_pipeline(
            n_samples=500,
            n_features=10,
            test_size=1.0,
        )


def test_invalid_max_depth_is_rejected():
    with pytest.raises(ValueError):
        run_overfitting_pipeline(
            n_samples=500,
            n_features=10,
            max_depth=0,
        )