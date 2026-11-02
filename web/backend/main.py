from __future__ import annotations

from fastapi import FastAPI, HTTPException

from pipediag.persistence import (
    delete_run,
    list_runs,
    load_run,
    save_run,
)

from pathlib import Path

from fastapi import Request
from fastapi.staticfiles import StaticFiles

from pipediag.application import DiagnosticRunResult
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.explanation import (
    EvidenceExplanation,
    ExplanationSpec,
    HypothesisExplanation,
)

from .schemas import DiagnosticRunResponse
from .serializers import diagnostic_run_result_to_response
from .integration import DiagnosticWebIntegration
from .service import DiagnosticRunService


import numpy as np

import pipediag
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipediag.application import (
    DiagnosticApplication,
    DiagnosticEvidenceProvider,
    DiagnosticRunResult,
)
from pipediag.diagnostics.engine_factory import (
    build_calibrated_diagnostic_engine,
)
from pipediag.evidence.runtime_provider import RuntimeEvidenceProvider

app = FastAPI(
    title="PipeDiag API",
    description=(
        "Web API for the PipeDiag explainable machine learning "
        "pipeline diagnostic framework."
    ),
    version="0.1.0",
)



@app.middleware("http")
async def handle_api_prefix(request: Request, call_next):
    """
    Support /api-prefixed frontend requests in production.

    The Vite development proxy already removes this prefix.
    """
    path = request.scope["path"]

    if path.startswith("/api/"):
        request.scope["path"] = path[4:]
        request.scope["raw_path"] = request.scope["path"].encode()

    return await call_next(request)


run_service = DiagnosticRunService()
web_integration = DiagnosticWebIntegration(run_service)


def publish_diagnostic_run(
    result: DiagnosticRunResult,
    runtime_context=None,
) -> str:
    return web_integration.publish(
        result,
        runtime_context=runtime_context,
    )


@app.get("/runs")
def list_saved_diagnostic_runs() -> list[dict]:
    """List persisted diagnostic run summaries."""
    return list_runs()


@app.get(
    "/runs/{run_id}",
    response_model=DiagnosticRunResponse,
)
def get_diagnostic_run(run_id: str) -> DiagnosticRunResponse:
    # Preserve existing in-memory behavior.
    result = run_service.get(run_id)

    if result is not None:
        runtime_context = run_service.get_runtime_context(run_id)

        return diagnostic_run_result_to_response(
            result,
            runtime_context=runtime_context,
        )

    # Fall back to persistent JSON storage.
    try:
        record = load_run(run_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Diagnostic run '{run_id}' was not found.",
        )

    # Ensure the run ID is available to the frontend.
    record = dict(record)
    diagnostic = dict(record.get("diagnostic", {}))
    metadata = dict(diagnostic.get("metadata", {}))
    metadata["run_id"] = run_id
    diagnostic["metadata"] = metadata
    record["diagnostic"] = diagnostic

    try:
        return DiagnosticRunResponse.model_validate(record)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored diagnostic run has an invalid response format.",
        ) from exc




@app.delete("/runs/{run_id}")
def delete_diagnostic_run(run_id: str) -> dict[str, bool]:
    """Delete a run from in-memory and persistent storage."""
    try:
        persistent_deleted = delete_run(run_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    memory_deleted = run_service.delete(run_id)

    if not persistent_deleted and not memory_deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Diagnostic run '{run_id}' was not found.",
        )

    return {"deleted": True}



@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "pipediag",
    }


@app.get("/demo", response_model=DiagnosticRunResponse)
def demo_diagnostic() -> DiagnosticRunResponse:
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
            "run_id": "demo-run",
            "source": "web-demo",
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

    web_integration.publish(result)

    return diagnostic_run_result_to_response(
        result,
        run_type="demo",
    )








@app.post("/example-run", response_model=DiagnosticRunResponse)
def run_example_pipeline() -> DiagnosticRunResponse:
    """Execute a real, instrumented example pipeline and publish its result."""

    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [7.0, 8.0],
        [8.0, 7.0],
    ])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    application = DiagnosticApplication(
        evidence_provider=DiagnosticEvidenceProvider(
            RuntimeEvidenceProvider()
        ),
        diagnostic_engine=build_calibrated_diagnostic_engine(),
    )

    with pipediag.start(application=application) as run:
        run.instrument(pipeline)
        pipeline.fit(X_train, y_train)
        pipeline.predict(X_test)

        result = run.result(
            metadata={
                "run_id": run.run_id,
                "source": "web-example",
            }
        )

    publish_diagnostic_run(
        result,
        runtime_context=run.context,
    )


    save_run(
        result,
        runtime_context=run.context,
        run_type="example",
    )

    return diagnostic_run_result_to_response(
        result,
        runtime_context=run.context,
        run_type="example",
    )







# ------------------------------------------------------------
# Production React dashboard
# ------------------------------------------------------------

FRONTEND_DIST = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "dist"
)

if FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIST,
            html=True,
        ),
        name="dashboard",
    )