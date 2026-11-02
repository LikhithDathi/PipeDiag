import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipediag.runtime.context import RunContext
from pipediag.runtime.instrumentation import instrument_pipeline


def test_real_sklearn_pipeline_produces_runtime_events_and_evidence():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )

    context = RunContext()

    instrument_pipeline(pipeline, context)

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert predictions.shape == (100,)

    events = context.get_events()

    assert len(events) > 0

    operations = [event.operation for event in events]

    assert "fit" in operations
    assert "predict" in operations


def test_real_sklearn_pipeline_records_dataset_evidence():
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )

    context = RunContext()

    instrument_pipeline(pipeline, context)

    pipeline.fit(X, y)
    pipeline.predict(X)

    evidence_collector = context.evidence_collector

    assert evidence_collector is not None

    # The collector should have observed runtime activity.
    assert evidence_collector.lineage.count() > 0


def test_real_sklearn_pipeline_preserves_event_hierarchy():
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )

    context = RunContext()

    instrument_pipeline(pipeline, context)

    pipeline.fit(X, y)
    pipeline.predict(X)

    events = context.get_events()

    assert events

    # Every event must belong to this run.
    assert all(
        event.run_id == context.run_id
        for event in events
    )

    # Event IDs must be unique.
    event_ids = [event.event_id for event in events]

    assert len(event_ids) == len(set(event_ids))

    # Depth must never be negative.
    assert all(event.depth >= 0 for event in events)