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

    assert len(evidence) == 3

    item = evidence[0]
    leakage_item = evidence[1]

    evaluation_evidence = next(
        item for item in evidence
        if item.evidence_type == "evaluation_problem"
    )
    assert evaluation_evidence.status == EVIDENCE_INSUFFICIENT

    assert leakage_item.evidence_type == "leakage"
    assert leakage_item.source == "runtime"
    assert leakage_item.status == EVIDENCE_OBSERVED
    assert leakage_item.metadata["inference_eligible"] is False
    assert leakage_item.metadata["finding_count"] == 8

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

    assert len(evidence) == 3

    item = evidence[0]

    assert item.evidence_type == "execution_problem"
    assert item.status == EVIDENCE_INSUFFICIENT

    assert item.measurements == {}

    assert item.metadata["total_event_count"] == 0
    assert item.metadata["failed_event_count"] == 0
    assert item.metadata["failed_event_ids"] == []
    assert item.metadata["failed_operations"] == []


    leakage_item = evidence[1]

    assert leakage_item.evidence_type == "leakage"
    assert leakage_item.status == EVIDENCE_INSUFFICIENT
    assert leakage_item.metadata["inference_eligible"] is False
    assert leakage_item.metadata["findings"] == []


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











def make_g3_runtime_context(X_train, X_test, test_context="TEST"):
    from pipediag.evidence.lineage import DatasetLineage
    from pipediag.evidence.registry import DatasetRegistry
    from pipediag.evidence.store import DatasetContextRegistry
    from pipediag.evidence.models import FEATURE_DATA

    lineage = DatasetLineage()
    registry = DatasetRegistry()
    contexts = DatasetContextRegistry()

    source_ref = lineage.create_reference(
        role=FEATURE_DATA,
        object_id=id(object()),
    )

    train_ref = lineage.create_reference(
        shape=X_train.shape,
        role=FEATURE_DATA,
        parent_dataset_ids=[source_ref.dataset_id],
    )
    test_ref = lineage.create_reference(
        shape=X_test.shape,
        role=FEATURE_DATA,
        parent_dataset_ids=[source_ref.dataset_id],
    )

    registry.register(X_train, train_ref)
    registry.register(X_test, test_ref)

    contexts.set_context(train_ref.dataset_id, "TRAIN")
    contexts.set_context(test_ref.dataset_id, test_context)

    collector = SimpleNamespace(
        registry=registry,
        get_dataset_references=lambda: registry.datasets,
    )

    context = FakeRuntimeContext(events=[])
    context._events = []
    context._evaluations = []
    context.evidence_collector = collector
    context.dataset_contexts = contexts

    return context, train_ref, test_ref


def test_runtime_collects_calibrated_g3_measurement():
    import numpy as np

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 4))
    X_test = rng.normal(loc=0.5, size=(100, 4))

    context, train_ref, test_ref = make_g3_runtime_context(
        X_train, X_test
    )

    evidence = RuntimeEvidenceProvider().collect(context)
    g3 = next(e for e in evidence if e.evidence_type == "distribution_shift")

    expected = np.mean(
        np.abs(X_train.mean(axis=0) - X_test.mean(axis=0))
        / X_train.std(axis=0)
    )

    assert g3.status == EVIDENCE_OBSERVED
    assert g3.measurements["mean_absolute_standardized_shift"] == pytest.approx(expected)
    assert g3.measurements["value"] == pytest.approx(expected)
    assert g3.dataset_ids == [train_ref.dataset_id, test_ref.dataset_id]
    assert g3.contexts == ["TRAIN", "TEST"]
    assert g3.metadata["inference_eligible"] is True


def test_runtime_g3_returns_insufficient_without_train_test_pair():
    import numpy as np

    X_train = np.ones((20, 3))
    X_test = np.ones((10, 3))

    context, _, _ = make_g3_runtime_context(
        X_train, X_test, test_context="UNKNOWN"
    )

    evidence = RuntimeEvidenceProvider().collect(context)
    g3 = next(e for e in evidence if e.evidence_type == "distribution_shift")

    assert g3.status == EVIDENCE_INSUFFICIENT
    assert g3.metadata["inference_eligible"] is False


def test_runtime_g3_handles_constant_training_features():
    import numpy as np

    X_train = np.ones((20, 3))
    X_test = np.ones((10, 3))

    context, _, _ = make_g3_runtime_context(X_train, X_test)

    evidence = RuntimeEvidenceProvider().collect(context)
    g3 = next(e for e in evidence if e.evidence_type == "distribution_shift")

    assert g3.status == EVIDENCE_INSUFFICIENT
    assert g3.metadata["inference_eligible"] is False











def make_g4_runtime_context(y_train, train_context="TRAIN"):
    from pipediag.evidence.lineage import DatasetLineage
    from pipediag.evidence.registry import DatasetRegistry
    from pipediag.evidence.store import DatasetContextRegistry
    from pipediag.evidence.models import TARGET_DATA

    lineage = DatasetLineage()
    registry = DatasetRegistry()
    contexts = DatasetContextRegistry()

    source = object()
    source_ref = lineage.create_reference(
        role=TARGET_DATA,
        object_id=id(source),
    )

    train_ref = lineage.create_reference(
        shape=y_train.shape,
        role=TARGET_DATA,
        parent_dataset_ids=[source_ref.dataset_id],
    )

    registry.register(y_train, train_ref)
    contexts.set_context(train_ref.dataset_id, train_context)

    collector = SimpleNamespace(
        registry=registry,
        get_dataset_references=lambda: registry.datasets,
    )

    context = FakeRuntimeContext(events=[])
    context.evidence_collector = collector
    context.dataset_contexts = contexts

    return context, train_ref


def test_runtime_g4_collects_imbalanced_target_distribution():
    import numpy as np

    y_train = np.array([0] * 90 + [1] * 10)

    context, train_ref = make_g4_runtime_context(y_train)

    evidence = RuntimeEvidenceProvider().collect(context)

    g4 = next(
        item for item in evidence
        if item.evidence_type == "imbalance"
    )

    assert g4.status == EVIDENCE_OBSERVED
    assert g4.measurements["minority_class_proportion"] == pytest.approx(0.10)
    assert g4.measurements["value"] == pytest.approx(0.10)
    assert g4.measurements["imbalance_ratio"] == pytest.approx(9.0)
    assert g4.dataset_ids == [train_ref.dataset_id]
    assert g4.contexts == ["TRAIN"]
    assert g4.metadata["inference_eligible"] is True
    assert g4.metadata["metric"] == "minority_class_proportion"


def test_runtime_g4_collects_balanced_target_distribution():
    import numpy as np

    y_train = np.array([0] * 50 + [1] * 50)

    context, _ = make_g4_runtime_context(y_train)

    evidence = RuntimeEvidenceProvider().collect(context)

    g4 = next(
        item for item in evidence
        if item.evidence_type == "imbalance"
    )

    assert g4.status == EVIDENCE_OBSERVED
    assert g4.measurements["minority_class_proportion"] == pytest.approx(0.50)
    assert g4.measurements["value"] == pytest.approx(0.50)
    assert g4.measurements["imbalance_ratio"] == pytest.approx(1.0)
    assert g4.metadata["inference_eligible"] is True


def test_runtime_g4_returns_insufficient_without_train_target():
    import numpy as np

    y_train = np.array([0] * 90 + [1] * 10)

    context, _ = make_g4_runtime_context(
        y_train,
        train_context="UNKNOWN",
    )

    evidence = RuntimeEvidenceProvider().collect(context)

    g4 = next(
        item for item in evidence
        if item.evidence_type == "imbalance"
    )

    assert g4.status == EVIDENCE_INSUFFICIENT
    assert g4.metadata["inference_eligible"] is False







def make_g5_runtime_context(X_train, train_context="TRAIN"):
    from pipediag.evidence.lineage import DatasetLineage
    from pipediag.evidence.registry import DatasetRegistry
    from pipediag.evidence.store import DatasetContextRegistry
    from pipediag.evidence.models import FEATURE_DATA

    lineage = DatasetLineage()
    registry = DatasetRegistry()
    contexts = DatasetContextRegistry()

    source = object()

    train_ref = lineage.create_reference(
        shape=X_train.shape,
        role=FEATURE_DATA,
        object_id=id(source),
    )

    registry.register(X_train, train_ref)
    contexts.set_context(train_ref.dataset_id, train_context)

    collector = SimpleNamespace(
        registry=registry,
        get_dataset_references=lambda: registry.datasets,
    )

    context = FakeRuntimeContext(events=[])
    context.evidence_collector = collector
    context.dataset_contexts = contexts

    return context, train_ref


def test_runtime_g5_collects_missing_value_rate():
    import numpy as np

    X_train = np.array([
        [1.0, np.nan],
        [2.0, 3.0],
        [np.nan, 4.0],
        [5.0, 6.0],
    ])

    context, train_ref = make_g5_runtime_context(X_train)

    evidence = RuntimeEvidenceProvider().collect(context)

    g5 = next(
        item for item in evidence
        if item.evidence_type == "data_quality"
    )

    assert g5.status == EVIDENCE_OBSERVED
    assert g5.measurements["missing_value_rate"] == pytest.approx(0.25)
    assert g5.measurements["value"] == pytest.approx(0.25)
    assert g5.dataset_ids == [train_ref.dataset_id]
    assert g5.contexts == ["TRAIN"]
    assert g5.metadata["inference_eligible"] is True
    assert g5.metadata["metric"] == "missing_value_rate"


def test_runtime_g5_clean_data_has_zero_missing_rate():
    import numpy as np

    X_train = np.ones((20, 3))

    context, _ = make_g5_runtime_context(X_train)

    evidence = RuntimeEvidenceProvider().collect(context)

    g5 = next(
        item for item in evidence
        if item.evidence_type == "data_quality"
    )

    assert g5.status == EVIDENCE_OBSERVED
    assert g5.measurements["missing_value_rate"] == 0.0
    assert g5.metadata["inference_eligible"] is True


def test_runtime_g5_without_train_data_is_insufficient():
    import numpy as np

    X_train = np.array([
        [1.0, np.nan],
        [2.0, 3.0],
    ])

    context, _ = make_g5_runtime_context(
        X_train,
        train_context="UNKNOWN",
    )

    evidence = RuntimeEvidenceProvider().collect(context)

    g5 = next(
        item for item in evidence
        if item.evidence_type == "data_quality"
    )

    assert g5.status == EVIDENCE_INSUFFICIENT
    assert g5.metadata["inference_eligible"] is False
    assert g5.measurements == {}












def test_runtime_g7_detects_constant_features():
    import numpy as np

    X_train = np.array([
        [1.0, 0.0, 5.0, 2.0],
        [2.0, 0.0, 6.0, 2.0],
        [3.0, 0.0, 7.0, 2.0],
        [4.0, 0.0, 8.0, 2.0],
    ])

    context, train_ref = make_g5_runtime_context(X_train)

    evidence = RuntimeEvidenceProvider().collect(context)

    g7 = next(
        item for item in evidence
        if item.evidence_type == "feature_problem"
    )

    assert g7.status == EVIDENCE_OBSERVED
    assert g7.measurements["constant_feature_count"] == 2
    assert g7.measurements["n_features"] == 4
    assert g7.measurements["constant_feature_ratio"] == pytest.approx(0.5)
    assert g7.measurements["value"] == pytest.approx(0.5)
    assert g7.dataset_ids == [train_ref.dataset_id]
    assert g7.contexts == ["TRAIN"]
    assert g7.metadata["inference_eligible"] is True
    assert g7.metadata["metric"] == "constant_feature_ratio"


def test_runtime_g7_clean_features_have_zero_constant_ratio():
    import numpy as np

    X_train = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
    ])

    context, _ = make_g5_runtime_context(X_train)

    evidence = RuntimeEvidenceProvider().collect(context)

    g7 = next(
        item for item in evidence
        if item.evidence_type == "feature_problem"
    )

    assert g7.status == EVIDENCE_OBSERVED
    assert g7.measurements["constant_feature_count"] == 0
    assert g7.measurements["constant_feature_ratio"] == 0.0
    assert g7.metadata["inference_eligible"] is True


def test_runtime_g7_without_train_data_is_insufficient():
    import numpy as np

    X_train = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])

    context, _ = make_g5_runtime_context(
        X_train,
        train_context="TEST",
    )

    evidence = RuntimeEvidenceProvider().collect(context)

    g7 = next(
        item for item in evidence
        if item.evidence_type == "feature_problem"
    )

    assert g7.status == EVIDENCE_INSUFFICIENT
    assert g7.metadata["inference_eligible"] is False






def test_runtime_g3_ignores_constant_features_and_uses_variable_features():
    import numpy as np

    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(100, 4))
    X_test = rng.normal(loc=0.5, size=(50, 4))

    # Make the final training feature constant.
    X_train[:, 3] = 5.0
    X_test[:, 3] = 5.0

    context, _, _ = make_g3_runtime_context(X_train, X_test)

    evidence = RuntimeEvidenceProvider().collect(context)

    g3 = next(
        e for e in evidence
        if e.evidence_type == "distribution_shift"
    )

    expected = np.mean(
        np.abs(
            X_train[:, :3].mean(axis=0)
            - X_test[:, :3].mean(axis=0)
        )
        / X_train[:, :3].std(axis=0)
    )

    assert g3.status == EVIDENCE_OBSERVED
    assert g3.measurements["mean_absolute_standardized_shift"] == pytest.approx(
        expected
    )
    assert g3.metadata["inference_eligible"] is True