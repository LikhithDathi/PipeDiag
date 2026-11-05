import pytest
from types import SimpleNamespace

from pipediag.evidence.runtime_provider import RuntimeEvidenceProvider

from pipediag.diagnostics.evidence import (
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_OBSERVED,
)


class FakeEvidenceCollector:
    def get_dataset_references(self):
        return {
            "D-001": SimpleNamespace(dataset_id="D-001"),
            "D-002": SimpleNamespace(dataset_id="D-002"),
        }


class FakeRuntimeContext:
    def __init__(self):
        self._events = [
            SimpleNamespace(
                event_id="E-001",
                operation="fit",
                status="success",
            ),
        ]
        self._evaluations = [
            SimpleNamespace(
                evaluation_id="EV-001",
                metric_name="accuracy",
                metric_value=0.92,
                evaluation_context="TRAIN",
            ),
        ]
        self.evidence_collector = FakeEvidenceCollector()

    def get_events(self):
        return list(self._events)

    def get_evaluations(self):
        return list(self._evaluations)


def test_runtime_provider_accepts_run_context():
    context = FakeRuntimeContext()

    provider = RuntimeEvidenceProvider()

    evidence = provider.collect(context)

    assert len(evidence) == 4

    execution_item = evidence[0]
    leakage_item = evidence[1]
    generalization_item = evidence[2]
    evaluation_problem_item = evidence[3]



    assert leakage_item.evidence_type == "leakage"
    assert leakage_item.status == "OBSERVED"
    assert leakage_item.metadata["inference_eligible"] is False
    assert leakage_item.metadata["finding_count"] == 2

    assert evaluation_problem_item.evidence_type == "evaluation_problem"
    assert evaluation_problem_item.status == EVIDENCE_INSUFFICIENT
    assert evaluation_problem_item.metadata["inference_eligible"] is False

    assert execution_item.evidence_type == "execution_problem"
    assert execution_item.status == "OBSERVED"

    assert execution_item.measurements["value"] == 0.0
    assert execution_item.measurements["execution_failure_rate"] == 0.0

    assert generalization_item.evidence_type == "generalization"
    assert generalization_item.status == "INSUFFICIENT_EVIDENCE"
    assert "No TEST evaluation evidence" in generalization_item.reason


def test_runtime_provider_reads_existing_evaluations():
    context = FakeRuntimeContext()

    provider = RuntimeEvidenceProvider()

    evaluations = provider._get_evaluations(context)

    assert len(evaluations) == 1
    assert evaluations[0].evaluation_id == "EV-001"
    assert evaluations[0].metric_name == "accuracy"
    assert evaluations[0].metric_value == 0.92
    assert evaluations[0].evaluation_context == "TRAIN"




def test_runtime_provider_groups_evaluations_by_context():
    context = FakeRuntimeContext()

    context._evaluations = [
        SimpleNamespace(
            evaluation_id="EV-001",
            metric_name="accuracy",
            metric_value=0.95,
            evaluation_context="TRAIN",
        ),
        SimpleNamespace(
            evaluation_id="EV-002",
            metric_name="accuracy",
            metric_value=0.85,
            evaluation_context="TEST",
        ),
    ]

    provider = RuntimeEvidenceProvider()

    grouped = provider._get_evaluations_by_context(context)

    assert set(grouped) == {"TRAIN", "TEST"}
    assert len(grouped["TRAIN"]) == 1
    assert len(grouped["TEST"]) == 1
    assert grouped["TRAIN"][0].metric_value == 0.95
    assert grouped["TEST"][0].metric_value == 0.85






def test_runtime_provider_builds_generalization_evidence():
    context = FakeRuntimeContext()

    context._evaluations = [
        SimpleNamespace(
            evaluation_id="EV-001",
            metric_name="accuracy",
            metric_value=0.95,
            evaluation_context="TRAIN",
        ),
        SimpleNamespace(
            evaluation_id="EV-002",
            metric_name="accuracy",
            metric_value=0.85,
            evaluation_context="TEST",
        ),
    ]

    provider = RuntimeEvidenceProvider()

    evidence = provider._build_generalization_evidence(context)

    assert evidence.evidence_type == "generalization"
    assert evidence.status == "OBSERVED"
    assert evidence.measurements["gap"] == 0.10









def test_runtime_g1_does_not_change_bayesian_posterior():
    from pipediag.diagnostics.engine_factory import (
        build_calibrated_diagnostic_engine,
    )
    from pipediag.diagnostics.adapters import (
        diagnostic_evidence_to_group,
    )
    from pipediag.diagnostics.evidence import DiagnosticEvidence

    engine = build_calibrated_diagnostic_engine()

    baseline_evidence = DiagnosticEvidence(
        evidence_type="execution_problem",
        source="runtime",
        status="OBSERVED",
        measurements={
            "value": 0.0,
            "execution_failure_rate": 0.0,
        },
    )

    runtime_g1 = DiagnosticEvidence(
        evidence_type="leakage",
        source="runtime",
        status="OBSERVED",
        metadata={
            "inference_eligible": False,
            "finding_count": 1,
            "confirmed_count": 1,
            "findings": [
                {"status": "CONFIRMED_LEAKAGE"}
            ],
        },
    )

    baseline_group = diagnostic_evidence_to_group(
        baseline_evidence
    )
    g1_group = diagnostic_evidence_to_group(runtime_g1)

    baseline_result = engine.diagnose([baseline_group])
    result_with_g1 = engine.diagnose([
        baseline_group,
        g1_group,
    ])

    assert result_with_g1.posterior == baseline_result.posterior
    assert g1_group in result_with_g1.evidence
    assert g1_group.metadata["inference_eligible"] is False






