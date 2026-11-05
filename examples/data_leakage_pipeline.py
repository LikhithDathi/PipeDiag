import json
from pathlib import Path

import pipediag

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    print("=" * 70)
    print("INTENTIONAL DATA LEAKAGE EXPERIMENT")
    print("=" * 70)
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    # 1. Define the pipeline.
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

    # 2. Configure PipeDiag WITHOUT an LLM.
    evidence_provider = DiagnosticEvidenceProvider(
        RuntimeEvidenceProvider()
    )

    diagnostic_engine = build_calibrated_diagnostic_engine()

    diagnostic_application = DiagnosticApplication(
        evidence_provider=evidence_provider,
        diagnostic_engine=diagnostic_engine,
    )

    # 3. Execute under PipeDiag observation.
    with pipediag.start(
        application=diagnostic_application,
        metadata={
            "experiment": "intentional_data_leakage",
            "dataset": "sklearn_breast_cancer",
            "model": "LogisticRegression",
            "injected_issue": "test_set_used_during_training",
        },
    ) as run:

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

        print("\nTraining samples:", len(X_train))
        print("Testing samples:", len(X_test))

        # 4. Instrument the pipeline.
        pipeline = run.instrument(pipeline)

        # 5. Correct training operation.
        print("\nFitting pipeline on TRAIN data...")
        pipeline.fit(X_train, y_train)

        # 6. INTENTIONAL BUG:
        # The pipeline is fitted again using TEST data.
        # This contaminates the evaluation dataset.
        print("\n" + "!" * 70)
        print("INTENTIONAL BUG: Fitting pipeline on TEST data!")
        print("!" * 70)

        pipeline.fit(X_test, y_test)

        # 7. Evaluate the contaminated model.
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

        # 8. Acquire evaluation evidence.
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

        # 9. Run PipeDiag diagnostics.
        result = run.result()

    # 10. Save the diagnostic run.
    saved_run_id = save_run(
        result,
        runtime_context=run.context,
        run_type="data_leakage",
    )

    print(f"\nDashboard Run ID: {saved_run_id}")

    # 11. Display evaluation.
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy:  {test_accuracy:.4f}")
    print(f"F1 score:          {f1_score(y_test, test_predictions):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=dataset.target_names,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # 12. Display runtime observations.
    print("\n" + "=" * 70)
    print("RUNTIME OBSERVATIONS")
    print("=" * 70)

    for event in run.context.get_events():
        print(
            f"{event.event_id} | "
            f"{event.step_name} | "
            f"{event.component} | "
            f"{event.operation} | "
            f"{event.status}"
        )

    # 13. Display diagnostic evidence.
    diagnostic = result.diagnostic_result

    print("\n" + "=" * 70)
    print("DIAGNOSTIC EVIDENCE")
    print("=" * 70)

    for evidence in diagnostic.evidence:
        print(f"\nEvidence Group: {evidence.name}")
        print(f"Status: {evidence.status}")
        print(f"Value: {evidence.value}")

        if evidence.name == "G1":
            findings = evidence.metadata.get("findings", [])

            print(f"Total findings: {len(findings)}")

            for finding in findings:
                print(
                    f"  [{finding['status']}] "
                    f"{finding['leakage_type']} | "
                    f"{finding['operation']}"
                )
                print(f"  Reason: {finding['reason']}")

    # 14. Display deterministic diagnostic report.
    print("\n" + "=" * 70)
    print("PIPEDIAG DIAGNOSTIC REPORT")
    print("=" * 70)
    print(result.report)

    # 15. Save experiment results.
    output_dir = Path("artifacts/data_leakage_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "run_id": run.context.run_id,
        "dataset": "sklearn_breast_cancer",
        "model": "LogisticRegression",
        "injected_issue": "test_set_used_during_training",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "f1_score": float(f1_score(y_test, test_predictions)),
        },
        "runtime_event_count": run.context.event_count(),
        "diagnostics": {
            "uncertainty": float(diagnostic.uncertainty),
            "posterior": {
                hypothesis: float(probability)
                for hypothesis, probability in diagnostic.posterior.items()
            },
            "evidence": [
                {
                    "name": evidence.name,
                    "status": evidence.status,
                    "value": (
                        float(evidence.value)
                        if isinstance(evidence.value, (int, float))
                        else evidence.value
                    ),
                    "metadata": evidence.metadata,
                }
                for evidence in diagnostic.evidence
            ],
        },
        "diagnostic_report": str(result.report),
    }

    output_path = output_dir / "experiment_results.json"

    output_path.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8",
    )

    print(f"\nExperiment results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("DATA LEAKAGE EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
