from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class ImbalancePipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    accuracy: float
    train_class_counts: dict[int, int]
    test_class_counts: dict[int, int]
    train_class_proportions: dict[int, float]
    test_class_proportions: dict[int, float]
    minority_class: int
    minority_class_proportion: float
    imbalance_ratio: float
    random_state: int


def _validate_parameters(
    n_samples: int,
    n_features: int,
    minority_fraction: float,
    test_size: float,
) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not 0.0 < minority_fraction < 0.5:
        raise ValueError("minority_fraction must be between 0 and 0.5")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")


def _create_imbalanced_target(
    target: np.ndarray,
    minority_fraction: float,
    random_state: int,
) -> np.ndarray:
    if target.ndim != 1:
        raise ValueError("target must be one-dimensional")

    rng = np.random.default_rng(random_state)

    n_samples = target.shape[0]
    minority_count = max(1, int(round(n_samples * minority_fraction)))
    majority_count = n_samples - minority_count

    if majority_count <= minority_count:
        raise ValueError("minority_fraction must produce a strict minority class")

    minority_label = 1
    majority_label = 0

    indices = rng.permutation(n_samples)

    imbalanced_target = np.empty(n_samples, dtype=int)

    imbalanced_target[indices[:majority_count]] = majority_label
    imbalanced_target[indices[majority_count:]] = minority_label

    return imbalanced_target


def _class_counts(target: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(target, return_counts=True)

    return {
        int(label): int(count)
        for label, count in zip(unique, counts)
    }


def _class_proportions(target: np.ndarray) -> dict[int, float]:
    counts = _class_counts(target)
    total = target.shape[0]

    return {
        label: count / total
        for label, count in counts.items()
    }


def _minority_class(
    class_counts: dict[int, int],
) -> int:
    return min(class_counts, key=class_counts.get)


def run_imbalance_pipeline(
    n_samples: int,
    n_features: int,
    minority_fraction: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
) -> ImbalancePipelineResult:
    _validate_parameters(
        n_samples=n_samples,
        n_features=n_features,
        minority_fraction=minority_fraction,
        test_size=test_size,
    )

    dataset = create_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    imbalanced_target = _create_imbalanced_target(
        target=dataset.target,
        minority_fraction=minority_fraction,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        imbalanced_target,
        test_size=test_size,
        random_state=random_state,
        stratify=imbalanced_target,
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = float(
        accuracy_score(
            y_test,
            predictions,
        )
    )

    train_class_counts = _class_counts(y_train)
    test_class_counts = _class_counts(y_test)

    train_class_proportions = _class_proportions(y_train)
    test_class_proportions = _class_proportions(y_test)

    minority_class = _minority_class(train_class_counts)

    minority_class_proportion = float(
        train_class_proportions[minority_class]
    )

    majority_class_proportion = max(
        train_class_proportions.values()
    )

    imbalance_ratio = float(
        majority_class_proportion / minority_class_proportion
    )

    return ImbalancePipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        accuracy=accuracy,
        train_class_counts=train_class_counts,
        test_class_counts=test_class_counts,
        train_class_proportions=train_class_proportions,
        test_class_proportions=test_class_proportions,
        minority_class=minority_class,
        minority_class_proportion=minority_class_proportion,
        imbalance_ratio=imbalance_ratio,
        random_state=random_state,
    )