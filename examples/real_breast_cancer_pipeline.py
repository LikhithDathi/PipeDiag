
import json
from pathlib import Path

import pipediag

from pipediag.acquisition import (
    EvaluationAcquirer,
    SplitAcquirer,
    TRAIN,
    TEST,
)

from pipediag.llm.ollama import OllamaExplanationGenerator

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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

from pipediag.persistence import save_run

def main():
    # 1. Load a real scikit-learn classification dataset
    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    print("=" * 70)
    print("REAL BREAST CANCER CLASSIFICATION EXPERIMENT")
    print("=" * 70)

    print(f"Dataset: {dataset.DESCR.splitlines()[0]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {dataset.target_names.tolist()}")

    # 2. Define the actual ML pipeline
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

    # 3. Configure PipeDiag
    evidence_provider = DiagnosticEvidenceProvider(
        RuntimeEvidenceProvider()
    )

    diagnostic_engine = build_calibrated_diagnostic_engine()

    llm_generator = OllamaExplanationGenerator(
        model="phi3:latest",
    )

    diagnostic_application = DiagnosticApplication(
        evidence_provider=evidence_provider,
        diagnostic_engine=diagnostic_engine,
        explanation_generator=llm_generator,
    )

    # 4. Execute the pipeline under PipeDiag observation
    with pipediag.start(
        application=diagnostic_application,
        metadata={
            "experiment": "breast_cancer_classification",
            "dataset": "sklearn_breast_cancer",
            "model": "LogisticRegression",
        },
    ) as run:

        # 4.1 Split acquisition using the active run's shared registries.
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
        print("PIPEDIAG DATA SPLIT ACQUISITION")
        print("=" * 70)
        print(f"Method: {split_evidence.method}")
        print(f"Input samples: {split_evidence.result.n_input}")
        print(f"Training samples: {split_evidence.result.n_train}")
        print(f"Testing samples: {split_evidence.result.n_test}")
        print(
            f"Training ratio: "
            f"{split_evidence.result.effective_train_ratio:.4f}"
        )
        print(
            f"Testing ratio: "
            f"{split_evidence.result.effective_test_ratio:.4f}"
        )

        # 4.2 Instrument and execute the pipeline.
        pipeline = run.instrument(pipeline)

        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)
        predictions = pipeline.predict(X_test)

        # 4.3 Calculate training and testing accuracy.
        train_accuracy = accuracy_score(
            y_train,
            train_predictions,
        )

        test_accuracy = accuracy_score(
            y_test,
            predictions,
        )

        # 4.4 Acquire structured evaluation evidence.
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

        # 4.5 Run diagnosis after all evidence has been registered.
        result = run.result()

    # 4.6 Persist the completed diagnostic run for the dashboard.
    saved_run_id = save_run(
        result,
        runtime_context=run.context,
        run_type="breast_cancer",
    )

    print(f"\nDashboard Run ID: {saved_run_id}")


    # 5. Evaluate the actual model.
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
    }

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # 6. Display PipeDiag runtime observations.
    print("\n" + "=" * 70)
    print("PIPEDIAG RUNTIME OBSERVATIONS")
    print("=" * 70)

    print(f"Run ID: {run.run_id}")
    print(f"Runtime events: {run.context.event_count()}")

    for event in run.context.get_events():
        print(
            f"{event.event_id} | "
            f"{event.step_name} | "
            f"{event.component} | "
            f"{event.operation} | "
            f"{event.status} | "
            f"{event.duration_ms:.3f} ms"
        )

    # 7. Display acquired diagnostic evidence.
    diagnostic = result.diagnostic_result

    print("\n" + "=" * 70)
    print("PIPEDIAG DIAGNOSTIC EVIDENCE")
    print("=" * 70)

    for evidence in diagnostic.evidence:
        print(f"\nEvidence Group: {evidence.name}")
        print(f"Status: {evidence.status}")
        print(f"Value: {evidence.value}")

        measurements = evidence.metadata.get("measurements", {})

        for name, value in measurements.items():
            print(f"  {name}: {value}")

        if evidence.name == "G1":
            findings = evidence.metadata.get("findings", [])

            print(f"  Total findings: {len(findings)}")
            print(
                f"  Confirmed: "
                f"{evidence.metadata.get('confirmed_count', 0)}"
            )
            print(
                f"  Potential: "
                f"{evidence.metadata.get('potential_count', 0)}"
            )
            print(
                f"  Insufficient: "
                f"{evidence.metadata.get('insufficient_count', 0)}"
            )
            print(
                f"  No evidence: "
                f"{evidence.metadata.get('no_evidence_count', 0)}"
            )

            for finding in findings:
                print(
                    f"    [{finding['status']}] "
                    f"{finding['leakage_type']} | "
                    f"{finding['operation']}"
                )
                print(f"      Reason: {finding['reason']}")

    print(f"\nPosterior uncertainty: {diagnostic.uncertainty:.4f}")

    # 8. Compare deterministic and LLM-generated reports.

    from pipediag.diagnostics.report import DiagnosticReportRenderer

    deterministic_report = DiagnosticReportRenderer().render(
        result.explanation
    )

    llm_report = result.report

    print("\n" + "=" * 70)
    print("DETERMINISTIC DIAGNOSTIC REPORT")
    print("=" * 70)
    print(deterministic_report)

    print("\n" + "=" * 70)
    print("OLLAMA LLM-GENERATED DIAGNOSTIC REPORT")
    print("=" * 70)
    print(llm_report)

    print("\n" + "=" * 70)
    print("HYPOTHESIS EVIDENCE CLASSIFICATION")
    print("=" * 70)

    for hypothesis in result.explanation.hypotheses:
        print(
            f"{hypothesis.hypothesis_id} | "
            f"{hypothesis.title} | "
            f"Posterior: {hypothesis.posterior:.4f} | "
            f"Evidence supported: {hypothesis.evidence_supported}"
        )

    # 9. Save experiment results.
    output_dir = Path("artifacts/real_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "run_id": run.context.run_id,
        "dataset": "sklearn_breast_cancer",
        "model": "LogisticRegression",
        "samples": len(X),
        "features": X.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "split": {
            "method": split_evidence.method,
            "input_samples": split_evidence.result.n_input,
            "train_samples": split_evidence.result.n_train,
            "test_samples": split_evidence.result.n_test,
            "train_ratio": split_evidence.result.effective_train_ratio,
            "test_ratio": split_evidence.result.effective_test_ratio,
            "input_dataset_ids": split_evidence.input_dataset_ids,
            "output_dataset_ids": split_evidence.output_dataset_ids,
        },
        "metrics": {
            name: float(value)
            for name, value in metrics.items()
        },
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
        "runtime_event_count": run.context.event_count(),
        "diagnostics": {
            "uncertainty": float(diagnostic.uncertainty),
            "posterior": {
                hypothesis: float(probability)
                for hypothesis, probability
                in diagnostic.posterior.items()
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
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
