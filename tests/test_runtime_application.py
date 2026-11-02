from dataclasses import dataclass

import pytest

from pipediag.application import DiagnosticApplication
from pipediag.diagnostics.adapters import diagnostic_evidence_to_group
from pipediag.diagnostics.evidence import DiagnosticEvidence
from pipediag.runtime.context import RunContext


from pipediag.evidence.store import (
    TRAIN,
    TEST,
    UNKNOWN,
)



@dataclass
class FakeDiagnosticResult:
    posterior: dict
    uncertainty: float = 0.2


class FakeEvidenceProvider:
    def __init__(self, evidence):
        self.evidence = evidence

    def collect(self, source):
        return self.evidence


class FakeDiagnosticEngine:
    def __init__(self):
        self.received_evidence = None
        self.received_metadata = None

    def diagnose(self, evidence_groups, metadata=None):
        self.received_evidence = evidence_groups
        self.received_metadata = metadata

        return FakeDiagnosticResult(
            posterior={"H1": 0.8, "H2": 0.2},
        )


class FakeExplanationBuilder:
    def __init__(self):
        self.received_result = None

    def build(self, diagnostic_result):
        self.received_result = diagnostic_result
        return "Explanation"


class FakeReportRenderer:
    def __init__(self):
        self.received_result = None
        self.received_explanation = None

    def render(self, diagnostic_result, explanation=None):
        self.received_result = diagnostic_result
        self.received_explanation = explanation
        return "# Diagnostic Report"


def make_evidence(
    evidence_type="generalization",
    measurements=None,
    source="evaluation",
):
    return DiagnosticEvidence(
        evidence_type=evidence_type,
        source=source,
        measurements=measurements or {
            "train_value": 0.95,
            "test_value": 0.70,
            "gap": 0.25,
        },
    )


def test_diagnostic_evidence_can_be_converted_to_group():
    evidence = make_evidence()

    group = diagnostic_evidence_to_group(evidence)

    assert group.name == "G2"
    assert group.status == "observed"
    assert group.metadata["source"] == "evaluation"
    assert group.metadata["measurements"]["gap"] == 0.25


def test_application_accepts_normalized_evidence_groups():
    evidence = [
        make_evidence(
            evidence_type="generalization",
            measurements={
                "train_value": 0.95,
                "test_value": 0.70,
                "gap": 0.25,
            },
        ),
        make_evidence(
            evidence_type="imbalance",
            measurements={
                "class_count": 2,
                "majority_proportion": 0.9,
                "minority_proportion": 0.1,
                "imbalance_ratio": 9.0,
                "threshold": 5.0,
            },
            source="target_distribution",
        ),
    ]

    groups = [
        diagnostic_evidence_to_group(item)
        for item in evidence
    ]

    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=FakeEvidenceProvider(groups),
        diagnostic_engine=engine,
        explanation_builder=FakeExplanationBuilder(),
        report_renderer=FakeReportRenderer(),
    )

    result = application.run(source="fake-pipeline")

    assert result.report == "# Diagnostic Report"
    assert len(engine.received_evidence) == 2
    assert [item.name for item in engine.received_evidence] == ["G2", "G4"]


def test_unavailable_evidence_is_preserved_as_unavailable_group():
    evidence = DiagnosticEvidence(
        evidence_type="distribution_shift",
        source="feature_distribution",
        status="INSUFFICIENT_EVIDENCE",
        reason="Reference distribution unavailable",
    )

    group = diagnostic_evidence_to_group(evidence)

    assert group.name == "G3"
    assert group.status == "unavailable"
    assert group.metadata["reason"] == "Reference distribution unavailable"


def test_evidence_measurements_are_preserved_without_becoming_group_value():
    evidence = make_evidence(
        measurements={
            "gap": 0.25,
            "train_value": 0.95,
            "test_value": 0.70,
        }
    )

    group = diagnostic_evidence_to_group(evidence)

    assert group.value is None
    assert group.metadata["measurements"]["gap"] == 0.25
    assert group.metadata["measurements"]["train_value"] == 0.95
    assert group.metadata["measurements"]["test_value"] == 0.70


def test_application_rejects_none_evidence_from_provider():
    provider = FakeEvidenceProvider(None)

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=FakeDiagnosticEngine(),
    )

    with pytest.raises(ValueError):
        application.run(source="fake-pipeline")





def test_run_context_can_assign_dataset_context():
    context = RunContext()

    context.set_dataset_context(
        "D-001",
        TRAIN,
    )

    assert (
        context.get_dataset_context("D-001")
        == TRAIN
    )


def test_run_context_unknown_dataset_context_is_unknown():
    context = RunContext()

    assert (
        context.get_dataset_context("D-999")
        == UNKNOWN
    )


def test_run_context_can_assign_test_context():
    context = RunContext()

    context.set_dataset_context(
        "D-002",
        TEST,
    )

    assert (
        context.get_dataset_context("D-002")
        == TEST
    )




from pipediag.acquisition.evaluation import EvaluationEvidence
from pipediag.evidence.store import TRAIN, TEST


def test_run_context_stores_evaluation_evidence():
    context = RunContext()

    evaluation = EvaluationEvidence(
        metric_name="accuracy",
        metric_value=0.95,
        evaluation_context=TRAIN,
        sample_count=100,
        source="explicit_metric",
    )

    context.add_evaluation(evaluation)

    evaluations = context.get_evaluations()

    assert len(evaluations) == 1
    assert evaluations[0] is evaluation
    assert evaluations[0].metric_name == "accuracy"
    assert evaluations[0].metric_value == 0.95
    assert evaluations[0].evaluation_context == TRAIN


def test_run_context_returns_copy_of_evaluations():
    context = RunContext()

    evaluation = EvaluationEvidence(
        metric_name="accuracy",
        metric_value=0.90,
        evaluation_context=TEST,
        sample_count=50,
        source="explicit_metric",
    )

    context.add_evaluation(evaluation)

    evaluations = context.get_evaluations()
    evaluations.clear()

    assert len(context.get_evaluations()) == 1


def test_run_context_rejects_none_evaluation():
    context = RunContext()

    with pytest.raises(ValueError):
        context.add_evaluation(None)






def test_application_can_use_ollama_explanation_generator():
    from pipediag.llm.ollama import OllamaExplanationGenerator

    class FakeEvidenceProvider:
        def collect(self, source):
            return []

    class FakeDiagnosticEngine:
        def diagnose(self, evidence_groups, metadata=None):
            from pipediag.diagnostics.diagnostic_result import DiagnosticResult

            return DiagnosticResult(
                posterior={
                    "H1": 1 / 9,
                    "H2": 1 / 9,
                    "H3": 1 / 9,
                    "H4": 1 / 9,
                    "H5": 1 / 9,
                    "H6": 1 / 9,
                    "H7": 1 / 9,
                    "H8": 1 / 9,
                    "H9": 1 / 9,
                },
                uncertainty=1.0,
                evidence=[],
                metadata=metadata or {},
            )

    class FakeOllamaClient:
        def chat(self, **kwargs):
            assert kwargs["model"] == "phi3:latest"
            assert len(kwargs["messages"]) == 2

            return {
                "message": {
                    "content": (
                        "The available evidence does not support a sufficiently confident "
                        "diagnosis. Diagnostic uncertainty is 1.0 and the reporting threshold "
                        "is 0.20."
                    )
                }
            }

    generator = OllamaExplanationGenerator(
        model="phi3:latest",
        client=FakeOllamaClient(),
    )

    from pipediag.application import DiagnosticApplication

    application = DiagnosticApplication(
        evidence_provider=FakeEvidenceProvider(),
        diagnostic_engine=FakeDiagnosticEngine(),
        explanation_generator=generator,
    )

    result = application.run(source=None)

    assert result.report == (
        "The available evidence does not support a sufficiently confident "
        "diagnosis. Diagnostic uncertainty is 1.0 and the reporting threshold "
        "is 0.20."
    )

    assert result.explanation is not None
    assert result.diagnostic_result is not None