
import numpy as np

import pipediag

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def main():
    # ---------------------------------------------------------
    # 1. Prepare a small classification dataset
    # ---------------------------------------------------------

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

    y = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # ---------------------------------------------------------
    # 2. Define the ML pipeline
    # ---------------------------------------------------------

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    # ---------------------------------------------------------
    # 3. Configure the diagnostic application
    # ---------------------------------------------------------

    evidence_provider = DiagnosticEvidenceProvider(
        RuntimeEvidenceProvider()
    )

    diagnostic_engine = build_calibrated_diagnostic_engine()

    diagnostic_application = DiagnosticApplication(
        evidence_provider=evidence_provider,
        diagnostic_engine=diagnostic_engine,
    )

    # ---------------------------------------------------------
    # 4. Execute the pipeline under PipeDiag observation
    # ---------------------------------------------------------

    with pipediag.start(
        application=diagnostic_application,
    ) as run:

        pipeline = run.instrument(pipeline)

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        # -----------------------------------------------------
        # 5. Generate the diagnostic result
        # -----------------------------------------------------

        result = run.result()

    # ---------------------------------------------------------
    # 6. Display the results
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("                 PIPEDIAG DIAGNOSTIC REPORT")
    print("=" * 65)

    print(f"\nRun ID: {run.run_id}")
    print(f"Runtime events captured: {run.context.event_count()}")
    print(f"Predictions generated: {len(predictions)}")

    print("\n--- Runtime Events ---")

    for event in run.context.get_events():
        print(
            f"{event.event_id} | "
            f"depth={event.depth} | "
            f"parent={event.parent_event_id} | "
            f"{event.step_name} | "
            f"{event.component} | "
            f"{event.operation} | "
            f"{event.status} | "
            f"{event.duration_ms:.3f} ms"
        )


    # ---------------------------------------------------------
    # 7. Display acquired diagnostic evidence
    # ---------------------------------------------------------

    print("\n--- Acquired Evidence ---")

    diagnostic = result.diagnostic_result

    for evidence in diagnostic.evidence:
        print(f"\nEvidence Group: {evidence.name}")
        print(f"Status: {evidence.status}")
        print(f"Value: {evidence.value}")

        measurements = evidence.metadata.get(
            "measurements",
            {},
        )

        if measurements:
            print("Measurements:")

            for name, value in measurements.items():
                print(f"  {name}: {value}")

    print(f"\nPosterior uncertainty: {diagnostic.uncertainty:.4f}")

    print("\n--- Diagnostic Report ---")
    print(result.report)


    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
