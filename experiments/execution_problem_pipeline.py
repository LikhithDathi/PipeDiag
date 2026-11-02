from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiments.dataset import create_dataset


@dataclass(frozen=True)
class ExecutionProblemPipelineResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    accuracy: float

    total_execution_events: int
    failed_execution_events: int
    execution_failure_rate: float

    fault_injected: bool
    random_state: int


def run_execution_problem_pipeline(
    *,
    n_samples: int = 500,
    n_features: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    inject_failure: bool = False,
    failure_event_count: int = 1,
    total_execution_events: int = 3,
) -> ExecutionProblemPipelineResult:
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

    if total_execution_events <= 0:
        raise ValueError(
            "total_execution_events must be positive"
        )

    if failure_event_count <= 0:
        raise ValueError(
            "failure_event_count must be positive"
        )

    if failure_event_count >= total_execution_events:
        raise ValueError(
            "failure_event_count must be smaller than "
            "total_execution_events"
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
        stratify=dataset.target,
        random_state=random_state,
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

    accuracy = float(
        np.mean(predictions == y_test)
    )

    failed_execution_events = (
        failure_event_count
        if inject_failure
        else 0
    )

    execution_failure_rate = (
        failed_execution_events
        / total_execution_events
    )

    return ExecutionProblemPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        accuracy=accuracy,
        total_execution_events=total_execution_events,
        failed_execution_events=failed_execution_events,
        execution_failure_rate=execution_failure_rate,
        fault_injected=inject_failure,
        random_state=random_state,
    )