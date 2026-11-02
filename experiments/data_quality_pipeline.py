from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class DataQualityPipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    accuracy: float
    missing_value_rate: float
    random_state: int


def _validate_parameters(
    n_samples: int,
    n_features: int,
    test_size: float,
    missing_rate: float,
) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1"
        )

    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(
            "missing_rate must be between 0 and 1"
        )


def _inject_missing_values(
    features: np.ndarray,
    missing_rate: float,
    random_state: int,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be 2-dimensional")

    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(
            "missing_rate must be between 0 and 1"
        )

    rng = np.random.default_rng(random_state)

    result = features.copy()

    total_values = result.size
    missing_count = int(
        round(total_values * missing_rate)
    )

    if missing_count == 0:
        return result

    flat_indices = rng.choice(
        total_values,
        size=missing_count,
        replace=False,
    )

    result.flat[flat_indices] = np.nan

    return result


def _missing_value_rate(features: np.ndarray) -> float:
    if features.ndim != 2:
        raise ValueError("features must be 2-dimensional")

    if features.size == 0:
        return 0.0

    return float(np.isnan(features).mean())


def run_data_quality_pipeline(
    n_samples: int = 500,
    n_features: int = 10,
    test_size: float = 0.2,
    missing_rate: float = 0.0,
    random_state: int = 42,
) -> DataQualityPipelineResult:

    _validate_parameters(
        n_samples=n_samples,
        n_features=n_features,
        test_size=test_size,
        missing_rate=missing_rate,
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

    X_train = _inject_missing_values(
        X_train,
        missing_rate=missing_rate,
        random_state=random_state,
    )

    X_test = _inject_missing_values(
        X_test,
        missing_rate=missing_rate,
        random_state=random_state + 1,
    )

    observed_missing_rate = (
        _missing_value_rate(X_train)
        + _missing_value_rate(X_test)
    ) / 2.0

    imputer = SimpleImputer(strategy="mean")

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    model.fit(
        X_train_imputed,
        y_train,
    )

    predictions = model.predict(
        X_test_imputed
    )

    accuracy = float(
        np.mean(predictions == y_test)
    )

    return DataQualityPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        accuracy=accuracy,
        missing_value_rate=observed_missing_rate,
        random_state=random_state,
    )