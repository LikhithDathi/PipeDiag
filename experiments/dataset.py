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
) -> BinaryClassificationDataset:
    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive"
        )

    if n_features <= 0:
        raise ValueError(
            "n_features must be positive"
        )

    features, target = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(
            1,
            min(n_features, n_features // 2),
        ),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
    )

    return BinaryClassificationDataset(
        features=features,
        target=target,
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )