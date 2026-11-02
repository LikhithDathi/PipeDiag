from pipediag.evidence import (
    FEATURE_DATA,
    PREDICTION_DATA,
    TARGET_DATA,
    UNKNOWN,
    DatasetLineage,
    DatasetRegistry,
)


def test_feature_dataset_reference_creation():
    lineage = DatasetLineage()

    reference = lineage.create_reference(
        shape=(800, 20),
        dtype="float64",
        role=FEATURE_DATA,
    )

    assert reference.dataset_id == "D-0001"
    assert reference.role == FEATURE_DATA
    assert reference.shape == (800, 20)
    assert reference.sample_count == 800
    assert reference.feature_count == 20
    assert reference.dtype == "float64"


def test_target_dataset_reference():
    lineage = DatasetLineage()

    reference = lineage.create_reference(
        shape=(800,),
        dtype="int64",
        role=TARGET_DATA,
    )

    assert reference.role == TARGET_DATA
    assert reference.sample_count == 800
    assert reference.feature_count is None


def test_prediction_dataset_reference():
    lineage = DatasetLineage()

    reference = lineage.create_reference(
        shape=(200,),
        dtype="int64",
        role=PREDICTION_DATA,
    )

    assert reference.role == PREDICTION_DATA
    assert reference.shape == (200,)


def test_unknown_role_is_supported():
    lineage = DatasetLineage()

    reference = lineage.create_reference(
        shape=(100, 5),
        role=UNKNOWN,
    )

    assert reference.role == UNKNOWN


def test_dataset_lineage_relationship():
    lineage = DatasetLineage()

    original = lineage.create_reference(
        shape=(1000, 20),
        dtype="float64",
        role=FEATURE_DATA,
    )

    transformed = lineage.create_reference(
        shape=(1000, 20),
        dtype="float64",
        role=FEATURE_DATA,
        parent_dataset_ids=[original.dataset_id],
    )

    assert transformed.parent_dataset_ids == [
        original.dataset_id
    ]

    assert lineage.parents_of(
        transformed.dataset_id
    ) == [original.dataset_id]


def test_registry_reuses_same_object_reference():
    registry = DatasetRegistry()
    lineage = DatasetLineage()

    data = [[1, 2], [3, 4]]

    reference = lineage.create_reference(
        shape=(2, 2),
        role=FEATURE_DATA,
        object_id=id(data),
    )

    registry.register(data, reference)

    retrieved = registry.get(data)

    assert retrieved is reference
    assert retrieved.dataset_id == reference.dataset_id


def test_registry_distinguishes_different_objects():
    registry = DatasetRegistry()
    lineage = DatasetLineage()

    first_data = [[1, 2]]
    second_data = [[1, 2]]

    first_reference = lineage.create_reference(
        shape=(1, 2),
        role=FEATURE_DATA,
        object_id=id(first_data),
    )

    second_reference = lineage.create_reference(
        shape=(1, 2),
        role=FEATURE_DATA,
        object_id=id(second_data),
    )

    registry.register(first_data, first_reference)
    registry.register(second_data, second_reference)

    assert registry.get(first_data) is first_reference
    assert registry.get(second_data) is second_reference
    assert first_reference.dataset_id != second_reference.dataset_id




def test_registry_retains_objects_for_identity_stability():
    registry = DatasetRegistry()
    lineage = DatasetLineage()

    data = [[1, 2], [3, 4]]

    reference = lineage.create_reference(
        shape=(2, 2),
        role=FEATURE_DATA,
        object_id=id(data),
    )

    registry.register(data, reference)

    assert registry.get(data) is reference
    assert id(data) in registry._objects
    assert registry._objects[id(data)] is data