import numpy as np
import pytest

from experiments.imbalance_pipeline import (
    ImbalancePipelineResult,
    run_imbalance_pipeline,
)


def test_returns_expected_result_type():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert isinstance(result, ImbalancePipelineResult)


def test_output_shapes_are_consistent():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert result.X_train.shape[0] == result.y_train.shape[0]
    assert result.X_test.shape[0] == result.y_test.shape[0]
    assert result.X_train.shape[1] == 10
    assert result.X_test.shape[1] == 10


def test_training_and_test_sets_are_non_empty():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert len(result.X_train) > 0
    assert len(result.X_test) > 0


def test_binary_target_is_preserved():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert set(np.unique(result.y_train)) == {0, 1}
    assert set(np.unique(result.y_test)) == {0, 1}


def test_training_class_counts_sum_to_training_size():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert sum(result.train_class_counts.values()) == len(result.y_train)


def test_test_class_counts_sum_to_test_size():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert sum(result.test_class_counts.values()) == len(result.y_test)


def test_training_class_proportions_sum_to_one():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert sum(result.train_class_proportions.values()) == pytest.approx(1.0)


def test_test_class_proportions_sum_to_one():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
    )

    assert sum(result.test_class_proportions.values()) == pytest.approx(1.0)


def test_requested_minority_fraction_is_approximately_respected():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    assert result.minority_class_proportion == pytest.approx(
        0.10,
        abs=0.01,
    )


def test_strong_imbalance_produces_large_imbalance_ratio():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    assert result.imbalance_ratio > 5.0


def test_minority_class_is_correctly_identified():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    minority_count = result.train_class_counts[result.minority_class]
    majority_count = max(result.train_class_counts.values())

    assert minority_count < majority_count


def test_accuracy_is_valid():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    assert 0.0 <= result.accuracy <= 1.0


def test_predictions_match_test_size():
    result = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    assert len(result.predictions) == len(result.y_test)


def test_same_random_state_is_reproducible():
    result_a = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    result_b = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    assert np.array_equal(result_a.y_train, result_b.y_train)
    assert np.array_equal(result_a.y_test, result_b.y_test)
    assert np.array_equal(result_a.predictions, result_b.predictions)
    assert result_a.accuracy == pytest.approx(result_b.accuracy)


def test_different_minority_fractions_change_class_distribution():
    balanced = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.40,
        random_state=42,
    )

    imbalanced = run_imbalance_pipeline(
        n_samples=500,
        n_features=10,
        minority_fraction=0.10,
        random_state=42,
    )

    assert (
        imbalanced.minority_class_proportion
        < balanced.minority_class_proportion
    )

    assert (
        imbalanced.imbalance_ratio
        > balanced.imbalance_ratio
    )


def test_invalid_sample_count_is_rejected():
    with pytest.raises(ValueError):
        run_imbalance_pipeline(
            n_samples=0,
            n_features=10,
        )


def test_invalid_feature_count_is_rejected():
    with pytest.raises(ValueError):
        run_imbalance_pipeline(
            n_samples=500,
            n_features=0,
        )


@pytest.mark.parametrize(
    "minority_fraction",
    [0.0, -0.1, 0.5, 0.6],
)
def test_invalid_minority_fraction_is_rejected(minority_fraction):
    with pytest.raises(ValueError):
        run_imbalance_pipeline(
            n_samples=500,
            n_features=10,
            minority_fraction=minority_fraction,
        )


@pytest.mark.parametrize(
    "test_size",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_test_size_is_rejected(test_size):
    with pytest.raises(ValueError):
        run_imbalance_pipeline(
            n_samples=500,
            n_features=10,
            test_size=test_size,
        )