import numpy as np

from pipediag.acquisition.split import SplitAcquirer
from pipediag.evidence import (
    FEATURE_DATA,
    PREDICTION_DATA,
    TARGET_DATA,
    UNKNOWN,
)

from pipediag.evidence.store import (
    TEST,
    TRAIN,
    UNKNOWN,
)


def test_split_x_only():
    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)

    outputs, evidence = acquirer.split(
        X,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_test = outputs

    assert X_train.shape[0] == 40
    assert X_test.shape[0] == 10

    assert evidence.result.n_input == 50
    assert evidence.result.n_train == 40
    assert evidence.result.n_test == 10

    assert evidence.configuration.test_size_requested == 0.2
    assert evidence.configuration.random_state == 42


def test_split_x_and_y_roles():
    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)
    y = np.array([0] * 25 + [1] * 25)

    outputs, evidence = acquirer.split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    assert len(outputs) == 4

    input_refs = [
        acquirer.registry.get(X),
        acquirer.registry.get(y),
    ]

    assert input_refs[0].role == FEATURE_DATA
    assert input_refs[1].role == TARGET_DATA

    assert evidence.configuration.stratify_requested is True


def test_split_creates_train_test_lineage():
    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)

    outputs, evidence = acquirer.split(
        X,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_test = outputs

    original = acquirer.registry.get(X)
    train_reference = acquirer.registry.get(X_train)
    test_reference = acquirer.registry.get(X_test)

    assert train_reference is not None
    assert test_reference is not None

    assert train_reference.parent_dataset_ids == [
        original.dataset_id
    ]

    assert test_reference.parent_dataset_ids == [
        original.dataset_id
    ]


def test_split_effective_ratios():
    acquirer = SplitAcquirer()

    X = np.arange(200).reshape(100, 2)

    _, evidence = acquirer.split(
        X,
        test_size=0.25,
        random_state=42,
    )

    assert evidence.result.n_input == 100
    assert evidence.result.n_train == 75
    assert evidence.result.n_test == 25

    assert evidence.result.effective_train_ratio == 0.75
    assert evidence.result.effective_test_ratio == 0.25


def test_binary_representativeness_is_assessed():
    acquirer = SplitAcquirer()

    X = np.arange(200).reshape(100, 2)
    y = np.array([0] * 70 + [1] * 30)

    _, evidence = acquirer.split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    representation = evidence.representativeness

    assert representation.status == "assessed"
    assert representation.metric == (
        "binary_distribution_difference"
    )

    assert representation.value == 0.0


def test_multiclass_representativeness_is_assessed():
    acquirer = SplitAcquirer()

    X = np.arange(300).reshape(150, 2)
    y = np.array(
        [0] * 50
        + [1] * 50
        + [2] * 50
    )

    _, evidence = acquirer.split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    representation = evidence.representativeness

    assert representation.status == "assessed"
    assert representation.metric == "total_variation_distance"
    assert representation.value == 0.0


def test_split_without_target_does_not_assess_distribution():
    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)

    _, evidence = acquirer.split(
        X,
        test_size=0.2,
        random_state=42,
    )

    assert (
        evidence.representativeness.status
        == "not_assessed"
    )


def test_additional_arrays_are_unknown():
    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)
    y = np.array([0] * 25 + [1] * 25)
    weights = np.ones(50)

    _, evidence = acquirer.split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=42,
    )

    additional_reference = acquirer.registry.get(weights)

    assert additional_reference.role == UNKNOWN


def test_train_size_can_define_effective_split():
    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)

    _, evidence = acquirer.split(
        X,
        train_size=0.8,
        random_state=42,
    )

    assert evidence.result.n_train == 40
    assert evidence.result.n_test == 10



def test_split_assigns_train_context():

    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)

    outputs, _ = acquirer.split(
        X,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_test = outputs

    train_reference = (
        acquirer.registry.get(X_train)
    )

    test_reference = (
        acquirer.registry.get(X_test)
    )

    assert (
        acquirer.context_registry.get_context(
            train_reference.dataset_id
        )
        == TRAIN
    )

    assert (
        acquirer.context_registry.get_context(
            test_reference.dataset_id
        )
        == TEST
    )


def test_split_assigns_context_to_target_datasets():

    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)

    y = np.array(
        [0] * 25
        + [1] * 25
    )

    outputs, _ = acquirer.split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = outputs

    y_train_reference = (
        acquirer.registry.get(y_train)
    )

    y_test_reference = (
        acquirer.registry.get(y_test)
    )

    assert (
        acquirer.context_registry.get_context(
            y_train_reference.dataset_id
        )
        == TRAIN
    )

    assert (
        acquirer.context_registry.get_context(
            y_test_reference.dataset_id
        )
        == TEST
    )


def test_original_dataset_context_remains_unknown():

    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)

    outputs, _ = acquirer.split(
        X,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_test = outputs

    original_reference = (
        acquirer.registry.get(X)
    )

    assert (
        acquirer.context_registry.get_context(
            original_reference.dataset_id
        )
        == UNKNOWN
    )


def test_additional_arrays_receive_train_test_context():

    acquirer = SplitAcquirer()

    X = np.arange(100).reshape(50, 2)

    y = np.array(
        [0] * 25
        + [1] * 25
    )

    weights = np.ones(50)

    outputs, _ = acquirer.split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=42,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        weights_train,
        weights_test,
    ) = outputs

    weights_train_reference = (
        acquirer.registry.get(weights_train)
    )

    weights_test_reference = (
        acquirer.registry.get(weights_test)
    )

    assert (
        acquirer.context_registry.get_context(
            weights_train_reference.dataset_id
        )
        == TRAIN
    )

    assert (
        acquirer.context_registry.get_context(
            weights_test_reference.dataset_id
        )
        == TEST
    )