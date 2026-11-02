from __future__ import annotations

from pipediag.application import DiagnosticRunResult
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.explanation import ExplanationSpec

from web.backend.integration import DiagnosticWebIntegration
from web.backend.run_store import DiagnosticRunStore
from web.backend.service import DiagnosticRunService

import pipediag

from pipediag.application import DiagnosticApplication

def make_result(run_id: str = "R-web-001") -> DiagnosticRunResult:
    diagnostic_result = DiagnosticResult(
        posterior={"H1": 1.0},
        uncertainty=0.0,
        evidence=[],
        metadata={"run_id": run_id},
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.0,
        hypotheses=(),
        overall_assessment="No dominant diagnostic explanation identified.",
    )

    return DiagnosticRunResult(
        diagnostic_result=diagnostic_result,
        explanation=explanation,
        report="Diagnostic report",
    )


def test_publish_completed_diagnostic_run():
    store = DiagnosticRunStore()
    service = DiagnosticRunService(store)
    integration = DiagnosticWebIntegration(service)

    result = make_result()

    run_id = integration.publish(result)

    assert run_id == "R-web-001"
    assert service.get("R-web-001") is result


def test_publish_requires_result():
    integration = DiagnosticWebIntegration()

    try:
        integration.publish(None)
    except ValueError as exc:
        assert str(exc) == "diagnostic result is required"
    else:
        raise AssertionError("Expected ValueError")