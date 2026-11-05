from __future__ import annotations

import pipediag

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipediag.acquisition import SplitAcquirer
from pipediag.diagnostics.calibration import CalibrationObservation
from pipediag.evidence.runtime_provider import RuntimeEvidenceProvider


def _collect_runtime_rate(
    random_state: int,
    clean_fit_count: int,
    inject_leakage: bool,
) -> float:
    dataset = load_breast_cancer()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=random_state,
            ),
        ),
    ])

    with pipediag.start() as run:
        split_acquirer = SplitAcquirer(
            lineage=run.context.evidence_collector.lineage,
            registry=run.context.evidence_collector.registry,
            context_registry=run.context.dataset_contexts,
        )

        (X_train, X_test, y_train, y_test), _ = (
            split_acquirer.split(
                dataset.data,
                dataset.target,
                test_size=0.20,
                random_state=random_state,
                stratify=dataset.target,
            )
        )

        pipeline = run.instrument(pipeline)

        # Repeated clean fits provide controlled variation in
        # the number of legitimate state-learning operations.
        for _ in range(clean_fit_count):
            pipeline.fit(X_train, y_train)

        if inject_leakage:
            # Deliberately fit on TEST data.
            pipeline.fit(X_test, y_test)

        evidence = RuntimeEvidenceProvider().collect(
            run.context
        )

    leakage = next(
        item
        for item in evidence
        if item.evidence_type == "leakage"
    )

    if (
        leakage.metadata["eligible_operation_count"] <= 0
        or leakage.metadata["unknown_context_count"] != 0
        or leakage.metadata["unclassified_operation_count"] != 0
    ):
        raise ValueError("Runtime calibration evidence is incomplete.")

    rate = leakage.metadata.get("runtime_contamination_rate")

    if rate is None:
        raise ValueError(
            "Runtime contamination rate could not be calculated."
        )

    return float(rate)


def collect_runtime_leakage_calibration_observations(
    random_states: tuple[int, ...] = tuple(range(1, 11)),
) -> list[CalibrationObservation]:

    observations = []

    for index, random_state in enumerate(random_states):
        clean_fit_count = index + 1

        clean_rate = _collect_runtime_rate(
            random_state=random_state,
            clean_fit_count=clean_fit_count,
            inject_leakage=False,
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G1",
                hypothesis="H1",
                hypothesis_true=False,
                value=clean_rate,
                measurement="runtime_contamination_rate",
            )
        )

        contaminated_rate = _collect_runtime_rate(
            random_state=random_state,
            clean_fit_count=clean_fit_count,
            inject_leakage=True,
        )

        observations.append(
            CalibrationObservation(
                evidence_group="G1",
                hypothesis="H1",
                hypothesis_true=True,
                value=contaminated_rate,
                measurement="runtime_contamination_rate",
            )
        )

    return observations