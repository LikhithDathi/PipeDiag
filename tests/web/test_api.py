from __future__ import annotations

from fastapi.testclient import TestClient

from web.backend.main import app

import pytest

from pipediag.persistence import save_run


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "pipediag",
    }


def test_demo_endpoint():
    response = client.get("/demo")

    assert response.status_code == 200

    body = response.json()

    assert "diagnostic" in body
    assert "explanation" in body
    assert "report" in body

    assert body["diagnostic"]["posterior"]["H2"] == 0.62
    assert body["diagnostic"]["uncertainty"] == 0.41

    assert body["explanation"]["threshold"] == 0.2
    assert body["explanation"]["inconclusive"] is False


def test_stored_demo_run_endpoint():
    # /demo stores demo-run in the in-memory run store.
    demo_response = client.get("/demo")

    assert demo_response.status_code == 200

    response = client.get("/runs/demo-run")

    assert response.status_code == 200

    body = response.json()

    assert body["diagnostic"]["metadata"]["run_id"] == "demo-run"
    assert body["diagnostic"]["posterior"]["H2"] == 0.62
    assert body["report"] == "Diagnostic report"


def test_unknown_run_returns_404():
    response = client.get("/runs/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Diagnostic run 'does-not-exist' was not found."
    }



def test_publish_diagnostic_run():
    from pipediag.diagnostics.diagnostic_result import DiagnosticResult
    from pipediag.diagnostics.explanation import ExplanationSpec
    from pipediag.application import DiagnosticRunResult
    from web.backend.main import publish_diagnostic_run

    diagnostic_result = DiagnosticResult(
        posterior={"H1": 1.0},
        uncertainty=0.0,
        evidence=[],
        metadata={"run_id": "R-api-publish"},
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.0,
        hypotheses=(),
        overall_assessment="No dominant diagnostic explanation identified.",
    )

    result = DiagnosticRunResult(
        diagnostic_result=diagnostic_result,
        explanation=explanation,
        report="Diagnostic report",
    )

    run_id = publish_diagnostic_run(result)

    assert run_id == "R-api-publish"

    response = client.get("/runs/R-api-publish")

    assert response.status_code == 200
    body = response.json()
    assert body["diagnostic"]["metadata"]["run_id"] == "R-api-publish"
    assert body["diagnostic"]["posterior"]["H1"] == 1.0
    assert body["report"] == "Diagnostic report"









def test_persisted_run_endpoint(tmp_path, monkeypatch):
    import pipediag.persistence as persistence

    monkeypatch.setattr(
        persistence,
        "_default_storage_dir",
        lambda: tmp_path,
    )

    from tests.test_persistence import make_result

    run_id = save_run(
        make_result(),
        storage_dir=tmp_path,
    )

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200

    body = response.json()

    assert body["diagnostic"]["metadata"]["run_id"] == run_id
    assert body["diagnostic"]["posterior"] == {"H1": 1.0}
    assert body["report"] == "Test diagnostic report"


def test_list_persisted_runs_endpoint(tmp_path, monkeypatch):
    import pipediag.persistence as persistence

    monkeypatch.setattr(
        persistence,
        "_default_storage_dir",
        lambda: tmp_path,
    )

    from tests.test_persistence import make_result

    run_id = save_run(
        make_result(),
        storage_dir=tmp_path,
    )

    response = client.get("/runs")

    assert response.status_code == 200
    assert run_id in {
        item["run_id"]
        for item in response.json()
    }


def test_delete_persisted_run_endpoint(tmp_path, monkeypatch):
    import pipediag.persistence as persistence

    monkeypatch.setattr(
        persistence,
        "_default_storage_dir",
        lambda: tmp_path,
    )

    from tests.test_persistence import make_result

    run_id = save_run(
        make_result(),
        storage_dir=tmp_path,
    )

    response = client.delete(f"/runs/{run_id}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert persistence.load_run(
        run_id,
        storage_dir=tmp_path,
    ) is None


def test_invalid_run_id_returns_400():
    response = client.get("/runs/../outside")

    assert response.status_code in (400, 404)