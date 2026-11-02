from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class LeakagePipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    accuracy: float
    leakage_feature_index: int
    random_state: int


def _create_leakage_target(
    target: np.ndarray,
    leakage_strength: float,
    random_state: int,
) -> np.ndarray:
    """
    Create a continuous target-derived feature with configurable
    leakage strength.

    A strength of 1.0 produces the exact target.
    Lower strengths progressively mix the target signal with
    independent noise.
    """
    if not 0.0 < leakage_strength <= 1.0:
        raise ValueError(
            "leakage_strength must be between 0 and 1"
        )

    if leakage_strength == 1.0:
        return target.astype(float).copy()

    rng = np.random.default_rng(random_state)

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=target.shape[0],
    )

    target_signal = target.astype(float)

    leakage_feature = (
        leakage_strength * target_signal
        + (1.0 - leakage_strength) * noise
    )

    return leakage_feature


def run_leakage_pipeline(
    n_samples: int,
    n_features: int,
    test_size: float = 0.2,
    random_state: int = 42,
    leakage_strength: float = 1.0,
) -> LeakagePipelineResult:

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

    if not 0.0 < leakage_strength <= 1.0:
        raise ValueError(
            "leakage_strength must be between 0 and 1"
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

    leakage_feature_index = X_train.shape[1]

    leakage_train = _create_leakage_target(
        target=y_train,
        leakage_strength=leakage_strength,
        random_state=random_state,
    )

    leakage_test = _create_leakage_target(
        target=y_test,
        leakage_strength=leakage_strength,
        random_state=random_state + 1,
    )

    X_train = np.column_stack(
        [
            X_train,
            leakage_train,
        ]
    )

    X_test = np.column_stack(
        [
            X_test,
            leakage_test,
        ]
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

    return LeakagePipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        accuracy=float(accuracy),
        leakage_feature_index=leakage_feature_index,
        random_state=random_state,
    )