from types import SimpleNamespace

import pytest

from pipediag.diagnostics.evidence import (
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_OBSERVED,
)
from pipediag.evidence.runtime_provider import (
    RuntimeEvidenceProvider,
)


class FakeRuntimeContext:
    def __init__(self, events):
        self._events = events

    def get_events(self):
        return list(self._events)


def make_event(
    event_id: str,
    operation: str,
    status: str,
):
    return SimpleNamespace(
        event_id=event_id,
        operation=operation,
        status=status,
    )


def test_collect_computes_execution_failure_rate():
    context = FakeRuntimeContext(
        [
            make_event("E-001", "fit", "success"),
            make_event("E-002", "transform", "success"),
            make_event("E-003", "predict", "failure"),
            make_event("E-004", "score", "success"),
        ]
    )

    provider = RuntimeEvidenceProvider()

    evidence = provider.collect(context)

    assert len(evidence) == 1

    item = evidence[0]

    assert item.evidence_type == "execution_problem"
    assert item.source == "runtime"
    assert item.status == EVIDENCE_OBSERVED

    assert item.measurements["execution_failure_rate"] == pytest.approx(
        0.25
    )

    assert item.measurements["total_event_count"] == 4
    assert item.measurements["failed_event_count"] == 1

    assert item.metadata["failed_event_ids"] == ["E-003"]
    assert item.metadata["failed_operations"] == ["predict"]


def test_collect_preserves_multiple_failed_events():
    context = FakeRuntimeContext(
        [
            make_event("E-001", "fit", "success"),
            make_event("E-002", "transform", "error"),
            make_event("E-003", "predict", "failure"),
            make_event("E-004", "score", "success"),
        ]
    )

    provider = RuntimeEvidenceProvider()

    evidence = provider.collect(context)

    item = evidence[0]

    assert item.measurements["execution_failure_rate"] == pytest.approx(
        0.50
    )

    assert item.measurements["failed_event_count"] == 2

    assert item.metadata["failed_event_ids"] == [
        "E-002",
        "E-003",
    ]

    assert item.metadata["failed_operations"] == [
        "transform",
        "predict",
    ]


def test_collect_with_no_events_returns_insufficient_evidence():
    context = FakeRuntimeContext([])

    provider = RuntimeEvidenceProvider()

    evidence = provider.collect(context)

    assert len(evidence) == 1

    item = evidence[0]

    assert item.evidence_type == "execution_problem"
    assert item.status == EVIDENCE_INSUFFICIENT

    assert item.measurements == {}

    assert item.metadata["total_event_count"] == 0
    assert item.metadata["failed_event_count"] == 0
    assert item.metadata["failed_event_ids"] == []
    assert item.metadata["failed_operations"] == []


def test_collect_requires_runtime_context():
    provider = RuntimeEvidenceProvider()

    with pytest.raises(
        ValueError,
        match="runtime evidence source is required",
    ):
        provider.collect(None)


def test_collect_requires_get_events():
    provider = RuntimeEvidenceProvider()

    with pytest.raises(
        TypeError,
        match="must provide get_events",
    ):
        provider.collect(object())


def test_collect_rejects_none_events():
    class InvalidRuntimeContext:
        def get_events(self):
            return None

    provider = RuntimeEvidenceProvider()

    with pytest.raises(
        ValueError,
        match="returned None for events",
    ):
        provider.collect(InvalidRuntimeContext())