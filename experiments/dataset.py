
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification


@dataclass(frozen=True)
class BinaryClassificationDataset:
    features: np.ndarray
    target: np.ndarray
    n_samples: int
    n_features: int
    random_state: int


def create_dataset(
    n_samples: int,
    n_features: int,
    random_state: int,
    n_informative: int | None = None,
    n_redundant: int = 0,
    class_sep: float = 1.0,
    flip_y: float = 0.01,
    weights: tuple[float, float] | None = None,
) -> BinaryClassificationDataset:

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if n_informative is None:
        n_informative = max(
            1,
            min(n_features, n_features // 2),
        )

    if n_informative <= 0:
        raise ValueError("n_informative must be positive")

    if n_redundant < 0:
        raise ValueError("n_redundant must be non-negative")

    if n_informative + n_redundant > n_features:
        raise ValueError(
            "Informative and redundant features cannot exceed "
            "the total feature count"
        )

    if class_sep <= 0:
        raise ValueError("class_sep must be positive")

    if not 0.0 <= flip_y <= 1.0:
        raise ValueError("flip_y must be between 0 and 1")

    if weights is not None:
        if len(weights) != 2:
            raise ValueError("weights must contain two class weights")

        if any(weight <= 0 for weight in weights):
            raise ValueError("class weights must be positive")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("class weights must sum to one")

    features, target = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=weights,
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=random_state,
    )

    return BinaryClassificationDataset(
        features=features,
        target=target,
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )
