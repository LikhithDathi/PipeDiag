
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiments.dataset import create_dataset


DEFAULT_MAX_WILSON_HALF_WIDTH = 0.05


@dataclass(frozen=True)
class EvaluationProblemPipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    accuracy: float
    evaluation_sample_count: int
    wilson_interval_half_width: float
    max_wilson_half_width: float
    evaluation_precision_adequate: bool
    random_state: int


def _validate_parameters(
    n_samples: int,
    n_features: int,
    test_size: float,
    max_wilson_half_width: float,
) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    if not 0.0 < max_wilson_half_width < 1.0:
        raise ValueError(
            "max_wilson_half_width must be between 0 and 1"
        )


def calculate_wilson_interval_half_width(
    accuracy: float,
    sample_count: int,
    confidence_z: float = 1.96,
) -> float:
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be between 0 and 1")

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    if confidence_z <= 0.0:
        raise ValueError("confidence_z must be positive")

    n = float(sample_count)
    z = float(confidence_z)

    denominator = 1.0 + (z * z / n)

    half_width = (
        z
        * math.sqrt(
            accuracy * (1.0 - accuracy) / n
            + (z * z) / (4.0 * n * n)
        )
        / denominator
    )

    return float(half_width)


def run_evaluation_problem_pipeline(
    n_samples: int = 500,
    n_features: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    max_wilson_half_width: float = DEFAULT_MAX_WILSON_HALF_WIDTH,
) -> EvaluationProblemPipelineResult:

    _validate_parameters(
        n_samples=n_samples,
        n_features=n_features,
        test_size=test_size,
        max_wilson_half_width=max_wilson_half_width,
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

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = float(np.mean(predictions == y_test))
    evaluation_sample_count = len(y_test)

    wilson_interval_half_width = (
        calculate_wilson_interval_half_width(
            accuracy=accuracy,
            sample_count=evaluation_sample_count,
        )
    )

    evaluation_precision_adequate = (
        wilson_interval_half_width <= max_wilson_half_width
    )

    return EvaluationProblemPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        accuracy=accuracy,
        evaluation_sample_count=evaluation_sample_count,
        wilson_interval_half_width=wilson_interval_half_width,
        max_wilson_half_width=max_wilson_half_width,
        evaluation_precision_adequate=evaluation_precision_adequate,
        random_state=random_state,
    )
