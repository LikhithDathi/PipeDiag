from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class DistributionShiftPipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    shift_magnitude: float
    random_state: int


def _validate_parameters(
    n_samples: int,
    n_features: int,
    test_size: float,
    shift_magnitude: float,
) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    if shift_magnitude < 0.0:
        raise ValueError("shift_magnitude must be non-negative")


def _split_dataset(
    features: np.ndarray,
    target: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    indices = rng.permutation(len(features))

    test_count = max(1, int(round(len(features) * test_size)))

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        features[train_indices],
        features[test_indices],
        target[train_indices],
        target[test_indices],
    )


def _apply_distribution_shift(
    X_test: np.ndarray,
    shift_magnitude: float,
) -> np.ndarray:
    if shift_magnitude == 0.0:
        return X_test.copy()

    return X_test + shift_magnitude


def _calculate_shift_score(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> float:
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be two-dimensional")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "X_train and X_test must have the same number of features"
        )

    train_means = np.mean(X_train, axis=0)
    test_means = np.mean(X_test, axis=0)

    train_stds = np.std(X_train, axis=0)

    valid_features = train_stds > 1e-12

    if not np.any(valid_features):
        return 0.0

    standardized_differences = (
        np.abs(train_means[valid_features] - test_means[valid_features])
        / train_stds[valid_features]
    )

    return float(np.mean(standardized_differences))


def run_distribution_shift_pipeline(
    n_samples: int,
    n_features: int,
    test_size: float = 0.20,
    shift_magnitude: float = 0.0,
    random_state: int = 42,
) -> DistributionShiftPipelineResult:
    _validate_parameters(
        n_samples=n_samples,
        n_features=n_features,
        test_size=test_size,
        shift_magnitude=shift_magnitude,
    )

    dataset = create_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = _split_dataset(
        features=dataset.features,
        target=dataset.target,
        test_size=test_size,
        random_state=random_state,
    )

    X_test = _apply_distribution_shift(
        X_test=X_test,
        shift_magnitude=shift_magnitude,
    )

    measured_shift = _calculate_shift_score(
        X_train=X_train,
        X_test=X_test,
    )

    return DistributionShiftPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        shift_magnitude=measured_shift,
        random_state=random_state,
    )