import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipediag import RunContext
from pipediag.runtime.instrumentation import instrument_pipeline


def test_pipeline_execution_is_captured():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [7.0, 8.0],
        [8.0, 7.0],
    ])

    y = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    context = RunContext()

    instrument_pipeline(
        pipeline,
        context,
    )

    pipeline.fit(X_train, y_train)
    pipeline.predict(X_test)

    events = context.get_events()

    assert len(events) >= 4

    operations = [event.operation for event in events]

    assert "fit_transform" in operations
    assert "fit" in operations
    assert "transform" in operations
    assert "predict" in operations

    event_ids = [event.event_id for event in events]

    assert len(event_ids) == len(set(event_ids))

    assert all(event.status == "success" for event in events)




def test_pipeline_execution_creates_dataset_evidence():
    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pipediag import RunContext
    from pipediag.runtime.instrumentation import instrument_pipeline

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    context = RunContext()

    instrument_pipeline(
        pipeline=pipeline,
        context=context,
    )

    pipeline.fit(X, y)
    pipeline.predict(X)

    events = context.get_events()

    assert len(events) > 0

    dataset_events = [
        event
        for event in events
        if event.input_dataset_refs
        or event.output_dataset_refs
    ]

    assert len(dataset_events) > 0

    assert all(
        isinstance(dataset_id, str)
        for event in dataset_events
        for dataset_id in (
            event.input_dataset_refs
            + event.output_dataset_refs
        )
    )

    datasets = (
        context.evidence_collector.lineage.get_all()
    )

    assert len(datasets) > 0


    transformed_datasets = [
        dataset
        for dataset in datasets
        if dataset.parent_dataset_ids
    ]

    assert len(transformed_datasets) > 0