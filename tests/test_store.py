from pipediag.evidence.store import (
    DatasetContextRegistry,
    TEST,
    TRAIN,
    UNKNOWN,
)


def test_context_registry_defaults_to_unknown():
    registry = DatasetContextRegistry()

    assert (
        registry.get_context("D-001")
        == UNKNOWN
    )


def test_context_registry_stores_train_context():
    registry = DatasetContextRegistry()

    registry.set_context(
        "D-001",
        TRAIN,
    )

    assert (
        registry.get_context("D-001")
        == TRAIN
    )


def test_context_registry_stores_test_context():
    registry = DatasetContextRegistry()

    registry.set_context(
        "D-002",
        TEST,
    )

    assert (
        registry.get_context("D-002")
        == TEST
    )