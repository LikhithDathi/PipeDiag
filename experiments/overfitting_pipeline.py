from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class OverfittingPipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    train_accuracy: float
    test_accuracy: float
    generalization_gap: float
    max_depth: int | None
    random_state: int


def run_overfitting_pipeline(
    n_samples: int,
    n_features: int,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int | None = None,
) -> OverfittingPipelineResult:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1"
        )

    if max_depth is not None and max_depth <= 0:
        raise ValueError(
            "max_depth must be positive or None"
        )

    dataset = create_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    test_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    generalization_gap = (
        train_accuracy - test_accuracy
    )

    return OverfittingPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        train_accuracy=float(train_accuracy),
        test_accuracy=float(test_accuracy),
        generalization_gap=float(generalization_gap),
        max_depth=max_depth,
        random_state=random_state,
    )