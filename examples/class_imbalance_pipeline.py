
import json
from pathlib import Path

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
    print("INTENTIONAL CLASS IMBALANCE EXPERIMENT")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Generate an imbalanced classification dataset.
    #
    # Class 0: approximately 95%
    # Class 1: approximately 5%
    # ------------------------------------------------------------

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.95, 0.05],
        flip_y=0.0,
        class_sep=1.0,
        random_state=42,
    )

    unique_classes, class_counts = __import__("numpy").unique(
        y,
        return_counts=True,
    )

    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    print("\nClass distribution:")

    for class_label, count in zip(unique_classes, class_counts):
        print(
            f"Class {class_label}: "
            f"{count} samples "
            f"({count / len(y):.2%})"
        )

    # ------------------------------------------------------------
    # 2. Configure the ML pipeline.
    #
    # No class balancing or resampling is applied.
    # This allows the experiment to preserve the imbalance.
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

    with pipediag.start(
        application=diagnostic_application,
        metadata={
            "experiment": "intentional_class_imbalance",
            "dataset": "synthetic_imbalanced_classification",
            "model": "LogisticRegression",
            "expected_evidence": "G4",
        },
    ) as run:

        # 4.1 Acquire train/test split.

        split_acquirer = SplitAcquirer(
            lineage=run.context.evidence_collector.lineage,
            registry=run.context.evidence_collector.registry,
            context_registry=run.context.dataset_contexts,
        )

        (X_train, X_test, y_train, y_test), split_evidence = (
            split_acquirer.split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )
        )

        print("\n" + "=" * 70)
        print("DATA SPLIT")
        print("=" * 70)

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Split method: {split_evidence.method}")

        # 4.2 Display training class distribution.

        import numpy as np

        train_classes, train_counts = np.unique(
            y_train,
            return_counts=True,
        )

        print("\nTraining class distribution:")

        for class_label, count in zip(
            train_classes,
            train_counts,
        ):
            print(
                f"Class {class_label}: "
                f"{count} ({count / len(y_train):.2%})"
            )

        # 4.3 Instrument and execute pipeline.

        pipeline = run.instrument(pipeline)

        print("\nFitting Logistic Regression...")

        pipeline.fit(X_train, y_train)

        # 4.4 Evaluate on train and test data.

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

        # 4.5 Acquire structured evaluation evidence.

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

        # 4.6 Run diagnosis.

        result = run.result()

    # ------------------------------------------------------------
    # 5. Persist diagnostic run.
    # ------------------------------------------------------------

    saved_run_id = save_run(
        result,
        runtime_context=run.context,
        run_type="class_imbalance",
    )

    print(f"\nDashboard Run ID: {saved_run_id}")

    # ------------------------------------------------------------
    # 6. Display model evaluation.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # ------------------------------------------------------------
    # 7. Display runtime observations.
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
    # 8. Display diagnostic evidence.
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
    # 9. Display deterministic diagnostic report.
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
    # 10. Display hypothesis classification.
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
    # 11. Save experiment results.
    # ------------------------------------------------------------

    output_dir = Path(
        "artifacts/diagnostic_experiments/G4"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "run_id": run.context.run_id,
        "experiment": "intentional_class_imbalance",
        "expected_evidence": "G4",
        "dataset": "synthetic_imbalanced_classification",
        "model": "LogisticRegression",
        "samples": len(X),
        "features": X.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": {
            str(label): int(count)
            for label, count in zip(
                unique_classes,
                class_counts,
            )
        },
        "metrics": {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
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
    print("CLASS IMBALANCE EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
