
import json
from pathlib import Path

import pipediag

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from pipediag.application import (
    DiagnosticApplication,
    DiagnosticEvidenceProvider,
)

from pipediag.diagnostics.engine_factory import (
    build_calibrated_diagnostic_engine,
)

from pipediag.evidence.runtime_provider import (
    RuntimeEvidenceProvider,
)

from pipediag.persistence import save_run


class IntentionalFailureTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Transformer that deliberately fails during transform.

    This is used only to test PipeDiag runtime failure detection.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise RuntimeError(
            "Intentional G8 execution failure for diagnostic testing."
        )


def main():

    print("=" * 70)
    print("INTENTIONAL RUNTIME EXECUTION FAILURE EXPERIMENT")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Generate a classification dataset.
    # ------------------------------------------------------------

    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42,
    )

    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    # ------------------------------------------------------------
    # 2. Configure a pipeline with an intentional failure.
    #
    # StandardScaler succeeds.
    # IntentionalFailureTransformer raises during transform.
    # LogisticRegression should never be reached.
    # ------------------------------------------------------------

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "intentional_failure",
            IntentionalFailureTransformer(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])

    # ------------------------------------------------------------
    # 3. Configure PipeDiag.
    # ------------------------------------------------------------

    evidence_provider = DiagnosticEvidenceProvider(
        RuntimeEvidenceProvider()
    )

    diagnostic_engine = build_calibrated_diagnostic_engine()

    diagnostic_application = DiagnosticApplication(
        evidence_provider=evidence_provider,
        diagnostic_engine=diagnostic_engine,
    )

    # ------------------------------------------------------------
    # 4. Execute under PipeDiag observation.
    # ------------------------------------------------------------

    expected_failure = False

    with pipediag.start(
        application=diagnostic_application,
        metadata={
            "experiment": "intentional_runtime_execution_failure",
            "dataset": "synthetic_execution_failure_classification",
            "model": "LogisticRegression",
            "expected_evidence": "G8",
        },
    ) as run:

        # 4.1 Instrument the pipeline.

        pipeline = run.instrument(pipeline)

        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION")
        print("=" * 70)

        try:

            pipeline.fit(X, y)

            print(
                "WARNING: Pipeline completed without the "
                "expected failure."
            )

        except RuntimeError as error:

            expected_failure = True

            print("\nExpected execution failure captured:")
            print(f"{type(error).__name__}: {error}")

        # 4.2 Run diagnosis after the failed operation.
        #
        # Do not execute another model operation after the
        # intentional failure.
        #
        # PipeDiag should use the runtime events already collected.

        result = run.result()

    # ------------------------------------------------------------
    # 5. Persist diagnostic run.
    # ------------------------------------------------------------

    saved_run_id = save_run(
        result,
        runtime_context=run.context,
        run_type="execution_failure",
    )

    print(f"\nDashboard Run ID: {saved_run_id}")

    # ------------------------------------------------------------
    # 6. Display runtime observations.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNTIME OBSERVATIONS")
    print("=" * 70)

    print(f"Runtime events: {run.context.event_count()}")

    for event in run.context.get_events():
        print(
            f"{event.event_id} | "
            f"{event.step_name} | "
            f"{event.component} | "
            f"{event.operation} | "
            f"{event.status}"
        )

    # ------------------------------------------------------------
    # 7. Display diagnostic evidence.
    # ------------------------------------------------------------

    diagnostic = result.diagnostic_result

    print("\n" + "=" * 70)
    print("DIAGNOSTIC EVIDENCE")
    print("=" * 70)

    for evidence in diagnostic.evidence:

        print(f"\nEvidence Group: {evidence.name}")
        print(f"Status: {evidence.status}")
        print(f"Value: {evidence.value}")

        measurements = evidence.metadata.get(
            "measurements",
            {},
        )

        for name, value in measurements.items():
            print(f"  {name}: {value}")

    # ------------------------------------------------------------
    # 8. Display deterministic diagnostic report.
    # ------------------------------------------------------------

    from pipediag.diagnostics.report import (
        DiagnosticReportRenderer,
    )

    report = DiagnosticReportRenderer().render(
        result.explanation
    )

    print("\n" + "=" * 70)
    print("PIPEDIAG DIAGNOSTIC REPORT")
    print("=" * 70)

    print(report)

    # ------------------------------------------------------------
    # 9. Display hypothesis classification.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYPOTHESIS EVIDENCE CLASSIFICATION")
    print("=" * 70)

    for hypothesis in result.explanation.hypotheses:
        print(
            f"{hypothesis.hypothesis_id} | "
            f"{hypothesis.title} | "
            f"Posterior: {hypothesis.posterior:.4f} | "
            f"Evidence supported: "
            f"{hypothesis.evidence_supported}"
        )

    # ------------------------------------------------------------
    # 10. Save experiment results.
    # ------------------------------------------------------------

    output_dir = Path(
        "artifacts/diagnostic_experiments/G8"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "run_id": run.context.run_id,
        "experiment": "intentional_runtime_execution_failure",
        "expected_evidence": "G8",
        "dataset": "synthetic_execution_failure_classification",
        "model": "LogisticRegression",
        "samples": len(X),
        "features": X.shape[1],
        "expected_failure_captured": expected_failure,
        "runtime_event_count": run.context.event_count(),
        "runtime_events": [
            {
                "event_id": event.event_id,
                "step_name": event.step_name,
                "component": event.component,
                "operation": event.operation,
                "status": event.status,
            }
            for event in run.context.get_events()
        ],
        "posterior": {
            hypothesis: float(probability)
            for hypothesis, probability
            in diagnostic.posterior.items()
        },
        "uncertainty": float(diagnostic.uncertainty),
        "hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "title": item.title,
                "posterior": float(item.posterior),
                "evidence_supported": item.evidence_supported,
            }
            for item in result.explanation.hypotheses
        ],
        "report": report,
    }

    output_path = output_dir / "experiment_results.json"

    output_path.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8",
    )

    print(f"\nExperiment results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("EXECUTION FAILURE EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
