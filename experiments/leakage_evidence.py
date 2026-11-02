from __future__ import annotations

import numpy as np

from experiments.clean_pipeline import CleanPipelineResult
from experiments.leakage_pipeline import LeakagePipelineResult
from pipediag.diagnostics.evidence_group import EvidenceGroup


def leakage_score(
    features: np.ndarray,
    target: np.ndarray,
    leakage_feature_index: int | None = None,
) -> float:
    """
    Measure the absolute feature-target correlation associated with leakage.

    When leakage_feature_index is provided, only the known injected leakage
    feature is measured. This prevents naturally predictive features in the
    underlying dataset from being mistaken for injected leakage.

    When leakage_feature_index is None, the maximum absolute feature-target
    correlation is retained for backwards compatibility.
    """
    if features.ndim != 2:
        raise ValueError(
            "features must be a 2-dimensional array"
        )

    if target.ndim != 1:
        raise ValueError(
            "target must be a 1-dimensional array"
        )

    if features.shape[0] == 0:
        raise ValueError(
            "features cannot be empty"
        )

    if features.shape[0] != target.shape[0]:
        raise ValueError(
            "features and target must have the same number "
            "of samples"
        )

    if leakage_feature_index is not None:
        if not 0 <= leakage_feature_index < features.shape[1]:
            raise ValueError(
                "leakage_feature_index must refer to a valid feature"
            )

        feature = features[:, leakage_feature_index]

        if np.std(feature) == 0.0:
            return 0.0

        correlation = np.corrcoef(
            feature,
            target,
        )[0, 1]

        if not np.isfinite(correlation):
            return 0.0

        return abs(float(correlation))

    correlations = []

    for index in range(features.shape[1]):
        feature = features[:, index]

        if np.std(feature) == 0.0:
            continue

        correlation = np.corrcoef(
            feature,
            target,
        )[0, 1]

        if np.isfinite(correlation):
            correlations.append(
                abs(float(correlation))
            )

    if not correlations:
        return 0.0

    return max(correlations)


def collect_leakage_evidence(
    result,
) -> EvidenceGroup:
    if result is None:
        raise ValueError(
            "Pipeline result cannot be None"
        )

    leakage_feature_index = getattr(
        result,
        "leakage_feature_index",
        None,
    )

    if leakage_feature_index is not None:
        score = leakage_score(
            result.X_train,
            result.y_train,
            leakage_feature_index=leakage_feature_index,
        )

        return EvidenceGroup(
            name="G1",
            value=score,
            status="observed",
            metadata={
                "metric": "leakage_feature_target_correlation",
                "source": "injected_leakage_feature",
                "feature_index": leakage_feature_index,
            },
        )

    # A clean pipeline contains no injected leakage feature.
    # Therefore, ordinary feature-target correlations must not be
    # interpreted as leakage evidence.
    return EvidenceGroup(
        name="G1",
        value=0.0,
        status="observed",
        metadata={
            "metric": "leakage_feature_target_correlation",
            "source": "no_injected_leakage_feature",
        },
    )