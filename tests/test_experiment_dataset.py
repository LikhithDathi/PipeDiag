import numpy as np
import pytest

from experiments.dataset import (
    BinaryClassificationDataset,
    create_dataset,
)


def test_dataset_contains_features_and_target():
    dataset = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    assert dataset.features.shape == (100, 5)
    assert dataset.target.shape == (100,)


def test_dataset_contains_binary_target():
    dataset = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    assert set(np.unique(dataset.target)).issubset(
        {0, 1}
    )


def test_dataset_has_expected_sample_count():
    dataset = create_dataset(
        n_samples=250,
        n_features=4,
        random_state=42,
    )

    assert len(dataset.features) == 250
    assert len(dataset.target) == 250


def test_dataset_is_reproducible():
    first = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    second = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    assert np.array_equal(
        first.features,
        second.features,
    )

    assert np.array_equal(
        first.target,
        second.target,
    )


def test_different_random_states_produce_different_data():
    first = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    second = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=99,
    )

    assert not np.array_equal(
        first.features,
        second.features,
    )


def test_dataset_stores_configuration():
    dataset = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    assert dataset.n_samples == 100
    assert dataset.n_features == 5
    assert dataset.random_state == 42


def test_dataset_rejects_invalid_sample_count():
    with pytest.raises(ValueError):
        create_dataset(
            n_samples=0,
            n_features=5,
            random_state=42,
        )


def test_dataset_rejects_invalid_feature_count():
    with pytest.raises(ValueError):
        create_dataset(
            n_samples=100,
            n_features=0,
            random_state=42,
        )


def test_dataset_rejects_negative_sample_count():
    with pytest.raises(ValueError):
        create_dataset(
            n_samples=-10,
            n_features=5,
            random_state=42,
        )


def test_dataset_rejects_negative_feature_count():
    with pytest.raises(ValueError):
        create_dataset(
            n_samples=100,
            n_features=-5,
            random_state=42,
        )


def test_dataset_is_immutable():
    dataset = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    with pytest.raises(AttributeError):
        dataset.n_samples = 200


def test_dataset_type_is_correct():
    dataset = create_dataset(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    assert isinstance(
        dataset,
        BinaryClassificationDataset,
    )