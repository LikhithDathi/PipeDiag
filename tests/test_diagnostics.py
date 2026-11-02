from pipediag.acquisition.evaluation import (
    EvaluationEvidence,
)

from pipediag.evidence.store import (
    DatasetContextRegistry,
    TRAIN,
    TEST,
    VALIDATION,
    UNKNOWN,
)

from pipediag.diagnostics.generalization import (
    GeneralizationDiagnostic,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    COMPUTED,
    INSUFFICIENT_EVIDENCE,
    NOT_COMPARABLE,
)

from pipediag.runtime.events import Event

from pipediag.evidence.models import (
    DatasetReference,
    FEATURE_DATA,
    TARGET_DATA,
)

from pipediag.diagnostics.leakage import (
    LeakageDiagnostic,
    TRAIN_TEST_CONTAMINATION,
    TARGET_FEATURE_DEPENDENCY,
    NO_EVIDENCE,
    POTENTIAL_LEAKAGE,
    CONFIRMED_LEAKAGE,
)


# ======================================================================
# Generalization helpers
# ======================================================================


def make_evaluation(
    metric_name: str,
    metric_value: float | None,
    context: str,
) -> EvaluationEvidence:

    return EvaluationEvidence(
        metric_name=metric_name,
        metric_value=metric_value,
        evaluation_context=context,
        feature_dataset_id="D-X",
        target_dataset_id="D-Y",
        source="ESTIMATOR_SCORE",
    )


# ======================================================================
# Generalization tests
# ======================================================================


def test_higher_is_better_gap_is_computed():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="accuracy",
        metric_value=0.98,
        context=TRAIN,
    )

    test = make_evaluation(
        metric_name="accuracy",
        metric_value=0.84,
        context=TEST,
    )

    result = diagnostic.analyze(
        evaluations=[train, test],
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == COMPUTED
    assert result.metric_name == "accuracy"
    assert result.train_value == 0.98
    assert result.test_value == 0.84
    assert result.gap == 0.14


def test_lower_is_better_gap_is_computed():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="rmse",
        metric_value=0.18,
        context=TRAIN,
    )

    test = make_evaluation(
        metric_name="rmse",
        metric_value=0.30,
        context=TEST,
    )

    result = diagnostic.analyze(
        evaluations=[train, test],
        metric_direction=LOWER_IS_BETTER,
    )

    assert result.status == COMPUTED
    assert result.gap == 0.12


def test_negative_gap_is_valid():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="accuracy",
        metric_value=0.82,
        context=TRAIN,
    )

    test = make_evaluation(
        metric_name="accuracy",
        metric_value=0.87,
        context=TEST,
    )

    result = diagnostic.analyze(
        evaluations=[train, test],
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == COMPUTED
    assert result.gap == -0.05


def test_missing_train_evidence_is_not_computed():

    diagnostic = GeneralizationDiagnostic()

    test = make_evaluation(
        metric_name="accuracy",
        metric_value=0.84,
        context=TEST,
    )

    result = diagnostic.analyze(
        evaluations=[test],
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.gap is None


def test_missing_test_evidence_is_not_computed():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="accuracy",
        metric_value=0.98,
        context=TRAIN,
    )

    result = diagnostic.analyze(
        evaluations=[train],
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.gap is None


def test_unknown_context_is_not_used():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="accuracy",
        metric_value=0.98,
        context=TRAIN,
    )

    unknown = make_evaluation(
        metric_name="accuracy",
        metric_value=0.84,
        context=UNKNOWN,
    )

    result = diagnostic.analyze(
        evaluations=[train, unknown],
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.gap is None


def test_validation_is_not_substituted_for_test():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="accuracy",
        metric_value=0.98,
        context=TRAIN,
    )

    validation = make_evaluation(
        metric_name="accuracy",
        metric_value=0.91,
        context=VALIDATION,
    )

    result = diagnostic.analyze(
        evaluations=[train, validation],
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.gap is None


def test_different_metrics_are_not_comparable():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="accuracy",
        metric_value=0.98,
        context=TRAIN,
    )

    test = make_evaluation(
        metric_name="f1",
        metric_value=0.84,
        context=TEST,
    )

    result = diagnostic.analyze(
        evaluations=[train, test],
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == NOT_COMPARABLE
    assert result.gap is None


def test_unknown_metric_direction_is_not_computed():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="accuracy",
        metric_value=0.98,
        context=TRAIN,
    )

    test = make_evaluation(
        metric_name="accuracy",
        metric_value=0.84,
        context=TEST,
    )

    result = diagnostic.analyze(
        evaluations=[train, test],
        metric_direction=UNKNOWN,
    )

    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.gap is None


def test_unavailable_metric_value_is_not_computed():

    diagnostic = GeneralizationDiagnostic()

    train = make_evaluation(
        metric_name="accuracy",
        metric_value=None,
        context=TRAIN,
    )

    test = make_evaluation(
        metric_name="accuracy",
        metric_value=0.84,
        context=TEST,
    )

    result = diagnostic.analyze(
        evaluations=[train, test],
        metric_direction=HIGHER_IS_BETTER,
    )

    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.gap is None


# ======================================================================
# Leakage helpers
# ======================================================================


def make_event(
    event_id: str,
    operation: str,
    input_dataset_refs: list[str],
    output_dataset_refs: list[str] | None = None,
) -> Event:

    return Event(
        event_id=event_id,
        run_id="R-001",
        operation=operation,
        input_dataset_refs=input_dataset_refs,
        output_dataset_refs=output_dataset_refs or [],
        status="success",
    )


def make_dataset(
    dataset_id: str,
    role: str,
) -> DatasetReference:

    return DatasetReference(
        dataset_id=dataset_id,
        role=role,
    )


def make_context_registry(
    contexts: dict[str, str],
) -> DatasetContextRegistry:

    registry = DatasetContextRegistry()

    for dataset_id, context in contexts.items():
        registry.set_context(
            dataset_id,
            context,
        )

    return registry


# ======================================================================
# Leakage tests
# ======================================================================


def test_train_fit_has_no_leakage():

    event = make_event(
        event_id="E-001",
        operation="fit",
        input_dataset_refs=["D-001"],
    )

    datasets = {
        "D-001": make_dataset(
            "D-001",
            FEATURE_DATA,
        )
    }

    registry = make_context_registry({
        "D-001": TRAIN,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    findings = [
        evidence
        for evidence in result
        if evidence.leakage_type == TRAIN_TEST_CONTAMINATION
    ]

    assert len(findings) == 1
    assert findings[0].status == NO_EVIDENCE


def test_test_fit_is_confirmed_leakage():

    event = make_event(
        event_id="E-002",
        operation="fit",
        input_dataset_refs=["D-002"],
    )

    datasets = {
        "D-002": make_dataset(
            "D-002",
            FEATURE_DATA,
        )
    }

    registry = make_context_registry({
        "D-002": TEST,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    finding = result[0]

    assert finding.status == CONFIRMED_LEAKAGE
    assert finding.leakage_type == TRAIN_TEST_CONTAMINATION
    assert finding.event_id == "E-002"
    assert finding.operation == "fit"
    assert finding.violating_dataset_ids == ["D-002"]


def test_train_and_test_fit_is_confirmed_leakage():

    event = make_event(
        event_id="E-003",
        operation="fit",
        input_dataset_refs=[
            "D-001",
            "D-002",
        ],
    )

    datasets = {
        "D-001": make_dataset(
            "D-001",
            FEATURE_DATA,
        ),
        "D-002": make_dataset(
            "D-002",
            FEATURE_DATA,
        ),
    }

    registry = make_context_registry({
        "D-001": TRAIN,
        "D-002": TEST,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    finding = result[0]

    assert finding.status == CONFIRMED_LEAKAGE
    assert finding.violating_dataset_ids == ["D-002"]


def test_train_fit_transform_has_no_leakage():

    event = make_event(
        event_id="E-004",
        operation="fit_transform",
        input_dataset_refs=["D-001"],
        output_dataset_refs=["D-003"],
    )

    datasets = {
        "D-001": make_dataset(
            "D-001",
            FEATURE_DATA,
        ),
        "D-003": make_dataset(
            "D-003",
            FEATURE_DATA,
        ),
    }

    registry = make_context_registry({
        "D-001": TRAIN,
        "D-003": TRAIN,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    findings = [
        evidence
        for evidence in result
        if evidence.leakage_type == TRAIN_TEST_CONTAMINATION
    ]

    assert len(findings) == 1
    assert findings[0].status == NO_EVIDENCE


def test_test_fit_transform_is_confirmed_leakage():

    event = make_event(
        event_id="E-005",
        operation="fit_transform",
        input_dataset_refs=["D-002"],
        output_dataset_refs=["D-004"],
    )

    datasets = {
        "D-002": make_dataset(
            "D-002",
            FEATURE_DATA,
        ),
        "D-004": make_dataset(
            "D-004",
            FEATURE_DATA,
        ),
    }

    registry = make_context_registry({
        "D-002": TEST,
        "D-004": TEST,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    finding = result[0]

    assert finding.status == CONFIRMED_LEAKAGE
    assert finding.violating_dataset_ids == ["D-002"]


def test_test_transform_has_no_leakage():

    event = make_event(
        event_id="E-006",
        operation="transform",
        input_dataset_refs=["D-002"],
        output_dataset_refs=["D-004"],
    )

    datasets = {
        "D-002": make_dataset(
            "D-002",
            FEATURE_DATA,
        ),
        "D-004": make_dataset(
            "D-004",
            FEATURE_DATA,
        ),
    }

    registry = make_context_registry({
        "D-002": TEST,
        "D-004": TEST,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    findings = [
        evidence
        for evidence in result
        if evidence.leakage_type == TRAIN_TEST_CONTAMINATION
    ]

    assert len(findings) == 1
    assert findings[0].status == NO_EVIDENCE


def test_unknown_context_produces_insufficient_evidence():

    event = make_event(
        event_id="E-007",
        operation="fit",
        input_dataset_refs=["D-005"],
    )

    datasets = {
        "D-005": make_dataset(
            "D-005",
            FEATURE_DATA,
        )
    }

    registry = make_context_registry({
        "D-005": UNKNOWN,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    finding = result[0]

    assert finding.status == INSUFFICIENT_EVIDENCE


def test_target_to_feature_dependency_is_potential_leakage():

    event = make_event(
        event_id="E-008",
        operation="transform",
        input_dataset_refs=["D-010"],
        output_dataset_refs=["D-011"],
    )

    datasets = {
        "D-010": make_dataset(
            "D-010",
            TARGET_DATA,
        ),
        "D-011": make_dataset(
            "D-011",
            FEATURE_DATA,
        ),
    }

    registry = make_context_registry({
        "D-010": TRAIN,
        "D-011": TRAIN,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    findings = [
        evidence
        for evidence in result
        if evidence.leakage_type == TARGET_FEATURE_DEPENDENCY
    ]

    assert len(findings) == 1

    finding = findings[0]

    assert finding.status == POTENTIAL_LEAKAGE
    assert finding.event_id == "E-008"
    assert finding.violating_dataset_ids == ["D-010"]


def test_feature_to_feature_dependency_has_no_target_leakage():

    event = make_event(
        event_id="E-009",
        operation="transform",
        input_dataset_refs=["D-020"],
        output_dataset_refs=["D-021"],
    )

    datasets = {
        "D-020": make_dataset(
            "D-020",
            FEATURE_DATA,
        ),
        "D-021": make_dataset(
            "D-021",
            FEATURE_DATA,
        ),
    }

    registry = make_context_registry({
        "D-020": TRAIN,
        "D-021": TRAIN,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    findings = [
        evidence
        for evidence in result
        if evidence.leakage_type == TARGET_FEATURE_DEPENDENCY
    ]

    assert len(findings) == 1
    assert findings[0].status == NO_EVIDENCE


def test_leakage_finding_preserves_traceability():

    event = make_event(
        event_id="E-010",
        operation="fit",
        input_dataset_refs=[
            "D-030",
            "D-031",
        ],
    )

    datasets = {
        "D-030": make_dataset(
            "D-030",
            FEATURE_DATA,
        ),
        "D-031": make_dataset(
            "D-031",
            FEATURE_DATA,
        ),
    }

    registry = make_context_registry({
        "D-030": TRAIN,
        "D-031": TEST,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    finding = result[0]

    assert finding.event_id == "E-010"
    assert finding.operation == "fit"

    assert finding.input_dataset_ids == [
        "D-030",
        "D-031",
    ]

    assert finding.output_dataset_ids == []

    assert finding.violating_dataset_ids == [
        "D-031",
    ]


def test_leakage_finding_preserves_rule():

    event = make_event(
        event_id="E-011",
        operation="fit",
        input_dataset_refs=["D-040"],
    )

    datasets = {
        "D-040": make_dataset(
            "D-040",
            FEATURE_DATA,
        )
    }

    registry = make_context_registry({
        "D-040": TEST,
    })

    diagnostic = LeakageDiagnostic(
        context_registry=registry,
    )

    result = diagnostic.analyze(
        events=[event],
        dataset_references=datasets,
    )

    finding = result[0]

    assert finding.rule != ""
    assert "TEST" in finding.rule
    assert "state" in finding.rule.lower()