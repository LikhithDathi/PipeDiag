from pipediag.acquisition.evaluation import (
    ESTIMATOR_SCORE,
    TEST,
    UNKNOWN,
    EvaluationAcquirer,
)

from pipediag.evidence.store import (
    DatasetContextRegistry,
    TRAIN,
    VALIDATION,
    TEST,
    UNKNOWN,
)


def test_score_evidence_is_created():
    acquirer = EvaluationAcquirer()

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.91,
        feature_dataset_id="D-001",
        target_dataset_id="D-002",
        evaluation_context=TEST,
        sample_count=100,
    )

    assert evidence.evidence_type == "evaluation"
    assert evidence.metric_name == "accuracy"
    assert evidence.metric_value == 0.91

    assert evidence.feature_dataset_id == "D-001"
    assert evidence.target_dataset_id == "D-002"

    assert evidence.evaluation_context == TEST
    assert evidence.sample_count == 100
    assert evidence.source == ESTIMATOR_SCORE


def test_evaluation_context_defaults_to_unknown():
    acquirer = EvaluationAcquirer()

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.85,
    )

    assert evidence.evaluation_context == UNKNOWN


def test_prediction_dataset_is_optional():
    acquirer = EvaluationAcquirer()

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.90,
        feature_dataset_id="D-001",
        target_dataset_id="D-002",
    )

    assert evidence.prediction_dataset_id is None


def test_metric_value_is_normalized_to_float():
    acquirer = EvaluationAcquirer()

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=1,
    )

    assert evidence.metric_value == 1.0
    assert isinstance(evidence.metric_value, float)


def test_invalid_metric_value_becomes_unavailable():
    acquirer = EvaluationAcquirer()

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value="not-a-number",
    )

    assert evidence.metric_value is None



def test_score_context_is_resolved_from_feature_and_target():

    registry = DatasetContextRegistry()

    registry.set_context(
        "D-001",
        TRAIN,
    )

    registry.set_context(
        "D-002",
        TRAIN,
    )

    acquirer = EvaluationAcquirer(
        context_registry=registry
    )

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.98,
        feature_dataset_id="D-001",
        target_dataset_id="D-002",
    )

    assert (
        evidence.evaluation_context
        == TRAIN
    )


def test_test_context_is_resolved():

    registry = DatasetContextRegistry()

    registry.set_context(
        "D-003",
        TEST,
    )

    registry.set_context(
        "D-004",
        TEST,
    )

    acquirer = EvaluationAcquirer(
        context_registry=registry
    )

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.84,
        feature_dataset_id="D-003",
        target_dataset_id="D-004",
    )

    assert (
        evidence.evaluation_context
        == TEST
    )


def test_validation_context_is_resolved():

    registry = DatasetContextRegistry()

    registry.set_context(
        "D-005",
        VALIDATION,
    )

    registry.set_context(
        "D-006",
        VALIDATION,
    )

    acquirer = EvaluationAcquirer(
        context_registry=registry
    )

    evidence = acquirer.acquire_score(
        metric_name="f1",
        metric_value=0.91,
        feature_dataset_id="D-005",
        target_dataset_id="D-006",
    )

    assert (
        evidence.evaluation_context
        == VALIDATION
    )


def test_mixed_contexts_resolve_to_unknown():

    registry = DatasetContextRegistry()

    registry.set_context(
        "D-001",
        TRAIN,
    )

    registry.set_context(
        "D-002",
        TEST,
    )

    acquirer = EvaluationAcquirer(
        context_registry=registry
    )

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.90,
        feature_dataset_id="D-001",
        target_dataset_id="D-002",
    )

    assert (
        evidence.evaluation_context
        == UNKNOWN
    )


def test_unknown_contexts_remain_unknown():

    registry = DatasetContextRegistry()

    acquirer = EvaluationAcquirer(
        context_registry=registry
    )

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.90,
        feature_dataset_id="D-001",
        target_dataset_id="D-002",
    )

    assert (
        evidence.evaluation_context
        == UNKNOWN
    )


def test_prediction_dataset_is_considered_for_context():

    registry = DatasetContextRegistry()

    registry.set_context(
        "D-001",
        TEST,
    )

    registry.set_context(
        "D-002",
        TEST,
    )

    registry.set_context(
        "D-003",
        TEST,
    )

    acquirer = EvaluationAcquirer(
        context_registry=registry
    )

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.88,
        feature_dataset_id="D-001",
        target_dataset_id="D-002",
        prediction_dataset_id="D-003",
    )

    assert (
        evidence.evaluation_context
        == TEST
    )


def test_explicit_context_overrides_registry():

    registry = DatasetContextRegistry()

    registry.set_context(
        "D-001",
        TRAIN,
    )

    registry.set_context(
        "D-002",
        TRAIN,
    )

    acquirer = EvaluationAcquirer(
        context_registry=registry
    )

    evidence = acquirer.acquire_score(
        metric_name="accuracy",
        metric_value=0.95,
        feature_dataset_id="D-001",
        target_dataset_id="D-002",
        evaluation_context=TEST,
    )

    assert (
        evidence.evaluation_context
        == TEST
    )