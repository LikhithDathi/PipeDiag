import numpy as np

from pipediag.evidence import (
    FEATURE_DATA,
    PREDICTION_DATA,
    TARGET_DATA,
    EvidenceCollector,
)


from pipediag.evidence.store import (
    DatasetContextRegistry,
    TRAIN,
    TEST,
    UNKNOWN,
)

def test_fit_maps_x_and_y_to_feature_and_target():
    collector = EvidenceCollector()

    X = np.zeros((100, 5))
    y = np.zeros(100)

    evidence = collector.observe(
        operation="fit",
        inputs=(X, y),
    )

    assert len(evidence["inputs"]) == 2
    assert len(evidence["outputs"]) == 0

    assert evidence["inputs"][0].role == FEATURE_DATA
    assert evidence["inputs"][1].role == TARGET_DATA

    assert evidence["inputs"][0].shape == (100, 5)
    assert evidence["inputs"][1].shape == (100,)


def test_fit_without_target_maps_only_feature():
    collector = EvidenceCollector()

    X = np.zeros((100, 5))

    evidence = collector.observe(
        operation="fit",
        inputs=(X,),
    )

    assert len(evidence["inputs"]) == 1
    assert evidence["inputs"][0].role == FEATURE_DATA


def test_transform_creates_feature_lineage():
    collector = EvidenceCollector()

    X = np.zeros((100, 5))
    X_transformed = np.ones((100, 5))

    evidence = collector.observe(
        operation="transform",
        inputs=(X,),
        output=X_transformed,
    )

    input_ref = evidence["inputs"][0]
    output_ref = evidence["outputs"][0]

    assert input_ref.role == FEATURE_DATA
    assert output_ref.role == FEATURE_DATA

    assert output_ref.parent_dataset_ids == [
        input_ref.dataset_id
    ]


def test_fit_transform_maps_feature_and_optional_target():
    collector = EvidenceCollector()

    X = np.zeros((100, 5))
    y = np.zeros(100)
    X_transformed = np.ones((100, 5))

    evidence = collector.observe(
        operation="fit_transform",
        inputs=(X, y),
        output=X_transformed,
    )

    assert evidence["inputs"][0].role == FEATURE_DATA
    assert evidence["inputs"][1].role == TARGET_DATA

    assert evidence["outputs"][0].role == FEATURE_DATA


def test_predict_maps_output_to_prediction():
    collector = EvidenceCollector()

    X = np.zeros((20, 5))
    predictions = np.zeros(20)

    evidence = collector.observe(
        operation="predict",
        inputs=(X,),
        output=predictions,
    )

    assert evidence["inputs"][0].role == FEATURE_DATA
    assert evidence["outputs"][0].role == PREDICTION_DATA

    assert evidence["outputs"][0].parent_dataset_ids == [
        evidence["inputs"][0].dataset_id
    ]


def test_predict_proba_maps_output_to_prediction():
    collector = EvidenceCollector()

    X = np.zeros((20, 5))
    probabilities = np.zeros((20, 2))

    evidence = collector.observe(
        operation="predict_proba",
        inputs=(X,),
        output=probabilities,
    )

    assert evidence["inputs"][0].role == FEATURE_DATA
    assert evidence["outputs"][0].role == PREDICTION_DATA


def test_decision_function_maps_output_to_prediction():
    collector = EvidenceCollector()

    X = np.zeros((20, 5))
    scores = np.zeros(20)

    evidence = collector.observe(
        operation="decision_function",
        inputs=(X,),
        output=scores,
    )

    assert evidence["inputs"][0].role == FEATURE_DATA
    assert evidence["outputs"][0].role == PREDICTION_DATA


def test_score_maps_x_and_y_without_dataset_output():
    collector = EvidenceCollector()

    X = np.zeros((20, 5))
    y = np.zeros(20)

    evidence = collector.observe(
        operation="score",
        inputs=(X, y),
        output=0.92,
    )

    assert len(evidence["inputs"]) == 2
    assert len(evidence["outputs"]) == 0

    assert evidence["inputs"][0].role == FEATURE_DATA
    assert evidence["inputs"][1].role == TARGET_DATA


def test_same_object_reuses_dataset_reference():
    collector = EvidenceCollector()

    X = np.zeros((100, 5))

    first = collector.observe(
        operation="transform",
        inputs=(X,),
        output=X,
    )

    second = collector.observe(
        operation="transform",
        inputs=(X,),
        output=X,
    )

    assert (
        first["inputs"][0].dataset_id
        == second["inputs"][0].dataset_id
    )

    assert (
        first["outputs"][0].dataset_id
        == second["outputs"][0].dataset_id
    )




def test_score_creates_evaluation_evidence_with_train_context():

    context_registry = DatasetContextRegistry()

    context_registry.set_context(
        "D-001",
        TRAIN,
    )

    context_registry.set_context(
        "D-002",
        TRAIN,
    )

    collector = EvidenceCollector(
        context_registry=context_registry,
    )

    X = object()
    y = object()

    # Register the exact objects first so the
    # collector obtains the expected dataset IDs.
    X_ref = collector._reference_for(
        X,
        role=FEATURE_DATA,
    )

    y_ref = collector._reference_for(
        y,
        role=TARGET_DATA,
    )

    context_registry.set_context(
        X_ref.dataset_id,
        TRAIN,
    )

    context_registry.set_context(
        y_ref.dataset_id,
        TRAIN,
    )

    result = collector.observe(
        operation="score",
        inputs=(X, y),
        output=0.98,
    )

    evaluation = result["evaluation"]

    assert evaluation is not None
    assert evaluation.metric_value == 0.98
    assert evaluation.evaluation_context == TRAIN
    assert (
        evaluation.feature_dataset_id
        == X_ref.dataset_id
    )
    assert (
        evaluation.target_dataset_id
        == y_ref.dataset_id
    )





def test_score_creates_evaluation_evidence_with_test_context():

    context_registry = DatasetContextRegistry()

    collector = EvidenceCollector(
        context_registry=context_registry,
    )

    X = object()
    y = object()

    X_ref = collector._reference_for(
        X,
        role=FEATURE_DATA,
    )

    y_ref = collector._reference_for(
        y,
        role=TARGET_DATA,
    )

    context_registry.set_context(
        X_ref.dataset_id,
        TEST,
    )

    context_registry.set_context(
        y_ref.dataset_id,
        TEST,
    )

    result = collector.observe(
        operation="score",
        inputs=(X, y),
        output=0.84,
    )

    evaluation = result["evaluation"]

    assert evaluation is not None
    assert evaluation.metric_value == 0.84
    assert evaluation.evaluation_context == TEST





def test_score_without_context_creates_unknown_evaluation():

    collector = EvidenceCollector()

    X = object()
    y = object()

    result = collector.observe(
        operation="score",
        inputs=(X, y),
        output=0.75,
    )

    evaluation = result["evaluation"]

    assert evaluation is not None
    assert evaluation.evaluation_context == UNKNOWN





def test_derived_dataset_inherits_train_context_through_lineage():
    from pipediag.evidence import FEATURE_DATA, TRAIN

    collector = EvidenceCollector()

    X_train = object()
    transformed = object()

    parent = collector._reference_for(
        X_train,
        role=FEATURE_DATA,
    )

    collector.context_registry.set_context(
        parent.dataset_id,
        TRAIN,
    )

    child = collector._reference_for(
        transformed,
        role=FEATURE_DATA,
        parent_dataset_ids=[
            parent.dataset_id,
        ],
    )

    assert collector._propagate_lineage_context(
        child.dataset_id
    ) == TRAIN

    assert (
        collector.context_registry.get_context(
            child.dataset_id
        )
        == TRAIN
    )


def test_derived_dataset_inherits_test_context_through_multiple_lineage_levels():
    from pipediag.evidence import FEATURE_DATA, TEST

    collector = EvidenceCollector()

    original = object()
    transformed_once = object()
    transformed_twice = object()

    root = collector._reference_for(
        original,
        role=FEATURE_DATA,
    )

    collector.context_registry.set_context(
        root.dataset_id,
        TEST,
    )

    first = collector._reference_for(
        transformed_once,
        role=FEATURE_DATA,
        parent_dataset_ids=[
            root.dataset_id,
        ],
    )

    second = collector._reference_for(
        transformed_twice,
        role=FEATURE_DATA,
        parent_dataset_ids=[
            first.dataset_id,
        ],
    )

    assert collector._propagate_lineage_context(
        second.dataset_id
    ) == TEST

    assert (
        collector.context_registry.get_context(
            second.dataset_id
        )
        == TEST
    )


def test_derived_dataset_with_conflicting_parent_contexts_remains_unknown():
    from pipediag.evidence import FEATURE_DATA, TRAIN, TEST, UNKNOWN

    collector = EvidenceCollector()

    train_data = object()
    test_data = object()
    derived = object()

    train_ref = collector._reference_for(
        train_data,
        role=FEATURE_DATA,
    )

    test_ref = collector._reference_for(
        test_data,
        role=FEATURE_DATA,
    )

    collector.context_registry.set_context(
        train_ref.dataset_id,
        TRAIN,
    )

    collector.context_registry.set_context(
        test_ref.dataset_id,
        TEST,
    )

    derived_ref = collector._reference_for(
        derived,
        role=FEATURE_DATA,
        parent_dataset_ids=[
            train_ref.dataset_id,
            test_ref.dataset_id,
        ],
    )

    assert collector._propagate_lineage_context(
        derived_ref.dataset_id
    ) == UNKNOWN

    assert (
        collector.context_registry.get_context(
            derived_ref.dataset_id
        )
        == UNKNOWN
    )


def test_derived_dataset_without_known_ancestor_context_remains_unknown():
    from pipediag.evidence import FEATURE_DATA, UNKNOWN

    collector = EvidenceCollector()

    original = object()
    derived = object()

    original_ref = collector._reference_for(
        original,
        role=FEATURE_DATA,
    )

    derived_ref = collector._reference_for(
        derived,
        role=FEATURE_DATA,
        parent_dataset_ids=[
            original_ref.dataset_id,
        ],
    )

    assert collector._propagate_lineage_context(
        derived_ref.dataset_id
    ) == UNKNOWN

    assert (
        collector.context_registry.get_context(
            derived_ref.dataset_id
        )
        == UNKNOWN
    )