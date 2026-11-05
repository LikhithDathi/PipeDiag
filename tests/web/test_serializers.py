from __future__ import annotations

from pipediag.application import DiagnosticRunResult
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.explanation import (
    EvidenceExplanation,
    ExplanationSpec,
    HypothesisExplanation,
)

from web.backend.serializers import diagnostic_run_result_to_response


def test_diagnostic_run_result_to_response():
    evidence = EvidenceGroup(
        name="G2",
        value=0.25,
        status="available",
        metadata={
            "metric": "generalization_gap",
        },
    )

    diagnostic_result = DiagnosticResult(
        posterior={
            "H1": 0.08,
            "H2": 0.62,
            "H3": 0.12,
            "H4": 0.04,
            "H5": 0.03,
            "H6": 0.02,
            "H7": 0.06,
            "H8": 0.02,
            "H9": 0.01,
        },
        uncertainty=0.41,
        evidence=[evidence],
        metadata={
            "run_id": "test-run",
        },
    )

    hypothesis = HypothesisExplanation(
        hypothesis_id="H2",
        title="Generalization Failure",
        posterior=0.62,
        evidence=(
            EvidenceExplanation(
                evidence_group="G2",
                metric="generalization_gap",
                value=0.25,
                interpretation=(
                    "The training and evaluation performance show "
                    "a substantial generalization gap."
                ),
            ),
        ),
        explanation_points=(
            "Generalization Failure is a potential explanation.",
        ),
        recommendation=(
            "Compare training and evaluation performance."
        ),
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.41,
        hypotheses=(hypothesis,),
        overall_assessment=(
            "Potential diagnostic issue(s) identified."
        ),
    )

    result = DiagnosticRunResult(
        diagnostic_result=diagnostic_result,
        explanation=explanation,
        report="Diagnostic report",
    )

    response = diagnostic_run_result_to_response(result)

    assert response.diagnostic.posterior["H2"] == 0.62
    assert response.diagnostic.uncertainty == 0.41
    assert response.diagnostic.metadata["run_id"] == "test-run"

    assert response.explanation.threshold == 0.20
    assert response.explanation.uncertainty == 0.41
    assert response.explanation.inconclusive is False

    assert len(response.explanation.hypotheses) == 1

    hypothesis_response = response.explanation.hypotheses[0]

    assert hypothesis_response.hypothesis_id == "H2"
    assert hypothesis_response.title == "Generalization Failure"
    assert hypothesis_response.posterior == 0.62

    assert len(hypothesis_response.evidence) == 1
    assert (
        hypothesis_response.evidence[0].metric
        == "generalization_gap"
    )

    assert response.report == "Diagnostic report"

def test_runtime_g1_findings_are_preserved_in_serialized_response():
    findings = [
        {
            "leakage_type": "train_test_contamination",
            "status": "CONFIRMED_LEAKAGE",
            "rule": "fit_on_test_data",
            "event_id": "event-1",
            "reason": "Estimator fitted using test data.",
        }
    ]

    runtime_g1 = EvidenceGroup(
        name="G1",
        value=None,
        status="observed",
        metadata={
            "source": "runtime",
            "reason": "Runtime leakage analysis completed.",
            "inference_eligible": False,
            "finding_count": 1,
            "confirmed_count": 1,
            "findings": findings,
        },
    )

    diagnostic_result = DiagnosticResult(
        posterior={
            "H1": 0.80,
            "H2": 0.02,
            "H3": 0.02,
            "H4": 0.02,
            "H5": 0.02,
            "H6": 0.02,
            "H7": 0.04,
            "H8": 0.03,
            "H9": 0.03,
        },
        uncertainty=0.3,
        evidence=[runtime_g1],
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.3,
        hypotheses=(),
        overall_assessment="Runtime evidence retained.",
    )

    result = DiagnosticRunResult(
        diagnostic_result=diagnostic_result,
        explanation=explanation,
        report="Diagnostic report",
    )

    response = diagnostic_run_result_to_response(result)

    assert len(response.evidence) == 1

    serialized = response.evidence[0]

    assert serialized.evidence_type == "G1"
    assert serialized.source == "runtime"
    assert serialized.status == "observed"
    assert serialized.metadata["inference_eligible"] is False
    assert serialized.metadata["finding_count"] == 1
    assert serialized.metadata["findings"] == findings
