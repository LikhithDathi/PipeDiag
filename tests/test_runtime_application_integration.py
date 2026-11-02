from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pipediag.application import (
    DiagnosticApplication,
    DiagnosticEvidenceProvider,
)
from pipediag.diagnostics.engine_factory import (
    build_calibrated_diagnostic_engine,
)
from pipediag.evidence.runtime_provider import RuntimeEvidenceProvider

import numpy as np

import pipediag

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class FakeRuntimeContext:
    def __init__(self, events):
        self._events = events

    def get_events(self):
        return list(self._events)


def make_event(event_id, operation, status):
    return SimpleNamespace(
        event_id=event_id,
        operation=operation,
        status=status,
    )


def test_runtime_evidence_reaches_diagnostic_application():
    project_root = Path(__file__).resolve().parents[1]

    artifact_path = (
        project_root
        / "artifacts"
        / "calibration"
        / "likelihood_models.json"
    )

    engine = build_calibrated_diagnostic_engine(
        artifact_path
    )

    provider = RuntimeEvidenceProvider()

    application = DiagnosticApplication(
        evidence_provider=DiagnosticEvidenceProvider(provider),
        diagnostic_engine=engine,
    )

    context = FakeRuntimeContext(
        [
            make_event("E-001", "fit", "success"),
            make_event("E-002", "transform", "success"),
            make_event("E-003", "predict", "failure"),
            make_event("E-004", "score", "success"),
        ]
    )

    result = application.run(context)

    assert result is not None

    diagnostic = result.diagnostic_result

    assert diagnostic is not None

    assert set(diagnostic.posterior) == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
    }

    assert abs(sum(diagnostic.posterior.values()) - 1.0) < 1e-9

    assert diagnostic.evidence

    g8 = [
        evidence
        for evidence in diagnostic.evidence
        if evidence.name == "G8"
    ]

    assert len(g8) == 1

    assert g8[0].metadata["metric"] == "execution_failure_rate"

    assert (
        g8[0].metadata["measurements"]["execution_failure_rate"]
        == 0.25
    )





def test_public_api_runs_real_pipeline_through_diagnostic_application():
    project_root = Path(__file__).resolve().parents[1]

    artifact_path = (
        project_root
        / "artifacts"
        / "calibration"
        / "likelihood_models.json"
    )

    application = DiagnosticApplication(
        evidence_provider=DiagnosticEvidenceProvider(
            RuntimeEvidenceProvider()
        ),
        diagnostic_engine=build_calibrated_diagnostic_engine(
            artifact_path
        ),
    )

    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 2.5],
            [2.5, 1.5],
            [6.0, 7.0],
            [7.0, 6.0],
            [6.5, 7.5],
            [7.5, 6.5],
        ]
    )

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    with pipediag.start(application=application) as run:
        instrumented_pipeline = run.instrument(pipeline)

        instrumented_pipeline.fit(X_train, y_train)
        predictions = instrumented_pipeline.predict(X_test)

        result = run.result()

    assert len(predictions) == len(y_test)

    assert run.run_id is not None
    assert run.context.get_events()

    assert result is not None
    assert result.diagnostic_result is not None
    assert result.explanation is not None
    assert result.report

    diagnostic = result.diagnostic_result

    assert set(diagnostic.posterior) == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
    }

    assert abs(sum(diagnostic.posterior.values()) - 1.0) < 1e-9

    g8 = [
        evidence
        for evidence in diagnostic.evidence
        if evidence.name == "G8"
    ]

    assert len(g8) == 1
    assert g8[0].status == "observed"