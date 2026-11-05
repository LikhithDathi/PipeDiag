
import json
from pathlib import Path

import numpy as np
import pipediag

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from pipediag.acquisition import (
    EvaluationAcquirer,
    SplitAcquirer,
    TRAIN,
    TEST,
)

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


def main():

    print("=" * 70)
    print("INTENTIONAL FEATURE DISTRIBUTION SHIFT EXPERIMENT")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Generate a classification dataset.
    # ------------------------------------------------------------

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        flip_y=0.02,
        class_sep=1.5,
        random_state=42,
    )

    # ------------------------------------------------------------
    # 2. Introduce a deliberate distribution shift.
    #
    # The first 20% retains the original distribution.
    # The remaining 80% receives a feature-value shift.
    #
    # The shift is applied before splitting so that the
    # split acquisition can observe the two distributions.
    # ------------------------------------------------------------

    split_index = int(len(X) * 0.20)

    X[split_index:, :] += 2.0

    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Distribution shift begins at sample: {split_index}")

    # ------------------------------------------------------------
    # 3. Configure the ML pipeline.
    # ------------------------------------------------------------

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])

    # ------------------------------------------------------------
    # 4. Configure PipeDiag.
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
    # 5. Execute under PipeDiag observation.
    # ------------------------------------------------------------

    with pipediag.start(
        application=diagnostic_application,
        metadata={
            "experiment": "intentional_feature_distribution_shift",
            "dataset": "synthetic_distribution_shift",
            "model": "LogisticRegression",
            "expected_evidence": "G3",
        },
    ) as run:

        # 5.1 Acquire the train/test split.

        split_acquirer = SplitAcquirer(
            lineage=run.context.evidence_collector.lineage,
            registry=run.context.evidence_collector.registry,
            context_registry=run.context.dataset_contexts,
        )

        (X_train, X_test, y_train, y_test), split_evidence = (
            split_acquirer.split(
                X,
                y,
                test_size=0.80,
                shuffle=False,
            )
        )

        print("\n" + "=" * 70)
        print("DATA SPLIT")
        print("=" * 70)

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Split method: {split_evidence.method}")

        # 5.2 Instrument and execute the pipeline.

        pipeline = run.instrument(pipeline)

        print("\nFitting pipeline on TRAIN data...")

        pipeline.fit(X_train, y_train)

        # 5.3 Evaluate on both datasets.

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)

        train_accuracy = accuracy_score(
            y_train,
            train_predictions,
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions,
        )

        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE")
        print("=" * 70)

        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy:  {test_accuracy:.4f}")
        print(
            f"Accuracy difference: "
            f"{train_accuracy - test_accuracy:.4f}"
        )

        # 5.4 Acquire structured evaluation evidence.

        evaluation_acquirer = EvaluationAcquirer()

        train_evidence = evaluation_acquirer.acquire_score(
            metric_name="accuracy",
            metric_value=train_accuracy,
            evaluation_context=TRAIN,
            sample_count=len(y_train),
        )

        test_evidence = evaluation_acquirer.acquire_score(
            metric_name="accuracy",
            metric_value=test_accuracy,
            evaluation_context=TEST,
            sample_count=len(y_test),
        )

        run.context.add_evaluation(train_evidence)
        run.context.add_evaluation(test_evidence)

        # 5.5 Run diagnosis after registering all evidence.

        result = run.result()

    # ------------------------------------------------------------
    # 6. Persist the diagnostic run.
    # ------------------------------------------------------------

    saved_run_id = save_run(
        result,
        runtime_context=run.context,
        run_type="distribution_shift",
    )

    print(f"\nDashboard Run ID: {saved_run_id}")

    # ------------------------------------------------------------
    # 7. Display model evaluation.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(classification_report(y_test, test_predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # ------------------------------------------------------------
    # 8. Display runtime observations.
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
    # 9. Display diagnostic evidence.
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

            if name == "feature_shifts" and isinstance(value, dict):
                print(f"  {name}:")

                for feature, shift in value.items():
                    print(f"    {feature}: {shift}")

            else:
                print(f"  {name}: {value}")

    # ------------------------------------------------------------
    # 10. Display the deterministic diagnostic report.
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
    # 11. Display hypothesis evidence classification.
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
    # 12. Save experiment results.
    # ------------------------------------------------------------

    output_dir = Path(
        "artifacts/diagnostic_experiments/G3"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "run_id": run.context.run_id,
        "experiment": "intentional_feature_distribution_shift",
        "expected_evidence": "G3",
        "dataset": "synthetic_distribution_shift",
        "model": "LogisticRegression",
        "samples": len(X),
        "features": X.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "accuracy_difference": float(
                train_accuracy - test_accuracy
            ),
        },
        "runtime_event_count": run.context.event_count(),
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
    print("DISTRIBUTION SHIFT EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
