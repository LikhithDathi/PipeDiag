from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class CleanPipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    accuracy: float
    random_state: int


def run_clean_pipeline(
    n_samples: int,
    n_features: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> CleanPipelineResult:

    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive"
        )

    if n_features <= 0:
        raise ValueError(
            "n_features must be positive"
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1"
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

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    return CleanPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        accuracy=float(accuracy),
        random_state=random_state,
    )