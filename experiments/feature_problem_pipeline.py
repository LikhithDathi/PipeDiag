from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class FeatureProblemPipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    accuracy: float

    constant_feature_count: int
    constant_feature_ratio: float
    n_features: int

    random_state: int


def _validate_parameters(
    n_samples: int,
    n_features: int,
    constant_feature_ratio: float,
) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not 0.0 <= constant_feature_ratio < 1.0:
        raise ValueError(
            "constant_feature_ratio must be between 0 and 1"
        )


def _calculate_constant_feature_ratio(
    features: np.ndarray,
) -> tuple[int, float]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")

    n_features = features.shape[1]

    if n_features == 0:
        raise ValueError("features must contain at least one feature")

    constant_feature_count = 0

    for feature_index in range(n_features):
        feature = features[:, feature_index]

        if np.allclose(feature, feature[0]):
            constant_feature_count += 1

    ratio = constant_feature_count / n_features

    return constant_feature_count, float(ratio)


def _inject_constant_features(
    features: np.ndarray,
    constant_feature_ratio: float,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")

    n_features = features.shape[1]

    constant_feature_count = int(
        round(n_features * constant_feature_ratio)
    )

    if constant_feature_count == 0:
        return features.copy()

    transformed = features.copy()

    for feature_index in range(
        n_features - constant_feature_count,
        n_features,
    ):
        transformed[:, feature_index] = 0.0

    return transformed


def run_feature_problem_pipeline(
    n_samples: int = 500,
    n_features: int = 10,
    constant_feature_ratio: float = 0.0,
    random_state: int = 42,
) -> FeatureProblemPipelineResult:

    _validate_parameters(
        n_samples=n_samples,
        n_features=n_features,
        constant_feature_ratio=constant_feature_ratio,
    )

    dataset = create_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    features = _inject_constant_features(
        dataset.features,
        constant_feature_ratio,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        dataset.target,
        test_size=0.20,
        random_state=random_state,
        stratify=dataset.target,
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = float(
        np.mean(predictions == y_test)
    )

    constant_feature_count, observed_ratio = (
        _calculate_constant_feature_ratio(
            features
        )
    )

    return FeatureProblemPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        accuracy=accuracy,
        constant_feature_count=constant_feature_count,
        constant_feature_ratio=observed_ratio,
        n_features=n_features,
        random_state=random_state,
    )