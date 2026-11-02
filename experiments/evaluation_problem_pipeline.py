from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class EvaluationProblemPipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    accuracy: float
    evaluation_sample_count: int
    evaluation_standard_error: float
    random_state: int


def _validate_parameters(
    n_samples: int,
    n_features: int,
    test_size: float,
) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1"
        )


def _calculate_accuracy_standard_error(
    accuracy: float,
    sample_count: int,
) -> float:
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(
            "accuracy must be between 0 and 1"
        )

    if sample_count <= 0:
        raise ValueError(
            "sample_count must be positive"
        )

    variance = (
        accuracy * (1.0 - accuracy)
        / sample_count
    )

    return float(math.sqrt(variance))


def run_evaluation_problem_pipeline(
    n_samples: int = 500,
    n_features: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> EvaluationProblemPipelineResult:

    _validate_parameters(
        n_samples=n_samples,
        n_features=n_features,
        test_size=test_size,
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

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    accuracy = float(
        np.mean(predictions == y_test)
    )

    evaluation_sample_count = len(y_test)

    evaluation_standard_error = (
        _calculate_accuracy_standard_error(
            accuracy=accuracy,
            sample_count=evaluation_sample_count,
        )
    )

    return EvaluationProblemPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        accuracy=accuracy,
        evaluation_sample_count=evaluation_sample_count,
        evaluation_standard_error=evaluation_standard_error,
        random_state=random_state,
    )