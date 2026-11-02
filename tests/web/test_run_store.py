from __future__ import annotations

from pipediag.application import DiagnosticRunResult
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.explanation import ExplanationSpec

from web.backend.run_store import DiagnosticRunStore


def make_result() -> DiagnosticRunResult:
    diagnostic_result = DiagnosticResult(
        posterior={
            "H1": 1.0,
        },
        uncertainty=0.0,
        evidence=[],
        metadata={
            "run_id": "R-test",
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


def test_store_save_and_get():
    store = DiagnosticRunStore()
    result = make_result()

    store.save("R-test", result)

    assert store.get("R-test") is result
    assert store.exists("R-test")
    assert store.count() == 1


def test_store_returns_none_for_unknown_run():
    store = DiagnosticRunStore()

    assert store.get("R-unknown") is None
    assert not store.exists("R-unknown")


def test_store_delete():
    store = DiagnosticRunStore()
    result = make_result()

    store.save("R-test", result)

    assert store.delete("R-test") is True
    assert store.get("R-test") is None
    assert store.count() == 0


def test_store_delete_unknown_run():
    store = DiagnosticRunStore()

    assert store.delete("R-unknown") is False


def test_store_clear():
    store = DiagnosticRunStore()

    store.save("R-1", make_result())
    store.save("R-2", make_result())

    assert store.count() == 2

    store.clear()

    assert store.count() == 0