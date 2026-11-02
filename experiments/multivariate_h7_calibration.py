
from __future__ import annotations

import numpy as np

from pipediag.diagnostics.multivariate_likelihood_model import (
    MultivariateGaussianLikelihoodModel,
)
from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import run_overfitting_pipeline


def _get_h7_features(
    random_state: int,
    max_depth: int,
) -> tuple[float, float]:
    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=random_state,
        max_depth=max_depth,
    )

    evidence = collect_generalization_evidence(result)
    measurements = evidence.metadata["measurements"]

    training_accuracy = float(
        measurements["training_accuracy"].value
    )
    generalization_gap = float(
        measurements["generalization_gap"].value
    )

    return training_accuracy, generalization_gap


def _estimate_distribution(
    observations: list[tuple[float, float]],
) -> tuple[
    tuple[float, float],
    tuple[tuple[float, float], tuple[float, float]],
]:
    matrix = np.asarray(observations, dtype=float)

    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("Expected a two-dimensional feature matrix")

    if len(matrix) < 3:
        raise ValueError(
            "At least three observations are required per class"
        )

    if not np.isfinite(matrix).all():
        raise ValueError("Calibration observations must be finite")

    mean = matrix.mean(axis=0)
    covariance = np.atleast_2d(
        np.cov(matrix, rowvar=False, ddof=1)
    )

    return (
        (float(mean[0]), float(mean[1])),
        (
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
    )


def build_multivariate_h7_model(
    random_states: tuple[int, ...] = tuple(range(1, 31)),
    normal_max_depths: tuple[int, ...] = (2, 3, 4, 5),
    underfitting_max_depth: int = 1,
    regularization: float = 1e-4,
) -> MultivariateGaussianLikelihoodModel:
    if len(random_states) < 3:
        raise ValueError("At least three random states are required")

    if len(set(random_states)) != len(random_states):
        raise ValueError("random_states must be unique")

    if not normal_max_depths:
        raise ValueError("normal_max_depths must not be empty")

    if any(depth <= 0 for depth in normal_max_depths):
        raise ValueError("Normal depths must be positive")

    if underfitting_max_depth <= 0:
        raise ValueError("underfitting_max_depth must be positive")

    if underfitting_max_depth in normal_max_depths:
        raise ValueError(
            "Underfitting depth must not be a normal depth"
        )

    positive: list[tuple[float, float]] = []
    negative: list[tuple[float, float]] = []

    for seed in random_states:
        positive.append(
            _get_h7_features(seed, underfitting_max_depth)
        )

        for depth in normal_max_depths:
            negative.append(_get_h7_features(seed, depth))

    mean_true, covariance_true = _estimate_distribution(positive)
    mean_false, covariance_false = _estimate_distribution(negative)

    return MultivariateGaussianLikelihoodModel(
        mean_true=mean_true,
        covariance_true=covariance_true,
        mean_false=mean_false,
        covariance_false=covariance_false,
        measurements=(
            "training_accuracy",
            "generalization_gap",
        ),
        regularization=regularization,
    )
