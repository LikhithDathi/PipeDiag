from pipediag import Event, RunContext


import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler

from pipediag import RunContext
from pipediag.runtime.instrumentation import instrument_estimator

from pipediag.evidence.store import (
    TRAIN,
)




def test_run_context_stores_events():
    context = RunContext()

    event = Event(
        event_id="E001",
        run_id=context.run_id,
        operation="fit",
        component="StandardScaler",
        step_name="scaler",
    )

    context.add_event(event)

    assert context.event_count() == 1

    stored_event = context.get_events()[0]

    assert stored_event.event_id == "E001"
    assert stored_event.run_id == context.run_id
    assert stored_event.operation == "fit"
    assert stored_event.component == "StandardScaler"





def test_failed_operation_is_recorded():
    scaler = StandardScaler()
    context = RunContext()

    instrument_estimator(
        estimator=scaler,
        context=context,
        step_name="scaler",
    )

    invalid_input = np.array([
        ["invalid", "data"],
        ["cannot", "scale"],
    ])

    with pytest.raises(ValueError):
        scaler.fit(invalid_input)

    events = context.get_events()

    assert len(events) == 1

    event = events[0]

    assert event.operation == "fit"
    assert event.status == "failed"
    assert event.metadata["error_type"] == "ValueError"
    assert "error_message" in event.metadata



def test_score_event_contains_evaluation_evidence():

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=40,
        n_features=4,
        random_state=42,
    )

    context = RunContext()

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    instrument_estimator(
        model,
        context,
    )

    # The dataset references are created by the
    # instrumented fit call.
    model.fit(X, y)

    # Find the feature and target references.
    feature_ref = (
        context.evidence_collector.registry.get(X)
    )

    target_ref = (
        context.evidence_collector.registry.get(y)
    )

    context.dataset_contexts.set_context(
        feature_ref.dataset_id,
        TRAIN,
    )

    context.dataset_contexts.set_context(
        target_ref.dataset_id,
        TRAIN,
    )

    score = model.score(X, y)

    score_event = next(
        event
        for event in context.events
        if event.operation == "score"
    )

    assert score_event.status == "success"

    assert (
        score_event.metadata["evaluation"]
        ["metric_value"]
        == score
    )

    assert (
        score_event.metadata["evaluation"]
        ["evaluation_context"]
        == TRAIN
    )