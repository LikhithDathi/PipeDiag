from __future__ import annotations

import pytest

from pipediag.application import DiagnosticRunResult
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.explanation import ExplanationSpec

from web.backend.run_store import DiagnosticRunStore
from web.backend.service import DiagnosticRunService


def make_result(
    run_id: str = "R-test",
) -> DiagnosticRunResult:
    diagnostic_result = DiagnosticResult(
        posterior={
            "H1": 1.0,
        },
        uncertainty=0.0,
        evidence=[],
        metadata={
            "run_id": run_id,
        },
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.0,
        hypotheses=(),
        overall_assessment=(
            "No dominant diagnostic explanation identified."
        ),
    )

    return DiagnosticRunResult(
        diagnostic_result=diagnostic_result,
        explanation=explanation,
        report="Diagnostic report",
    )


def test_submit_stores_completed_run():
    store = DiagnosticRunStore()
    service = DiagnosticRunService(store)

    result = make_result("R-001")

    run_id = service.submit(result)

    assert run_id == "R-001"
    assert service.get("R-001") is result


def test_submit_requires_run_id():
    store = DiagnosticRunStore()
    service = DiagnosticRunService(store)

    result = make_result("")
    result.diagnostic_result.metadata.pop("run_id")

    with pytest.raises(ValueError):
        service.submit(result)


def test_get_unknown_run_returns_none():
    service = DiagnosticRunService(
        DiagnosticRunStore()
    )

    assert service.get("R-unknown") is None


def test_delete_run():
    service = DiagnosticRunService(
        DiagnosticRunStore()
    )

    result = make_result("R-002")
    service.submit(result)

    assert service.delete("R-002") is True
    assert service.get("R-002") is None


def test_clear_runs():
    service = DiagnosticRunService(
        DiagnosticRunStore()
    )

    service.submit(make_result("R-003"))
    service.submit(make_result("R-004"))

    service.clear()

    assert service.get("R-003") is None
    assert service.get("R-004") is None