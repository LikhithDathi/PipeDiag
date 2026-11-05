
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
    print("INTENTIONAL EVALUATION UNCERTAINTY EXPERIMENT")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Generate a classification dataset.
    # ------------------------------------------------------------

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        flip_y=0.10,
        class_sep=0.8,
        random_state=42,
    )

    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    # ------------------------------------------------------------
    # 2. Configure the ML pipeline.
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
            "experiment": "intentional_evaluation_uncertainty",
            "dataset": "synthetic_small_evaluation_classification",
            "model": "LogisticRegression",
            "expected_evidence": "G6",
        },
    ) as run:

        # 4.1 Acquire train/test split.
        #
        # Only 20 samples are used for testing.
        # The remaining 980 samples are used for training.
        # --------------------------------------------------------

        split_acquirer = SplitAcquirer(
            lineage=run.context.evidence_collector.lineage,
            registry=run.context.evidence_collector.registry,
            context_registry=run.context.dataset_contexts,
        )

        (X_train, X_test, y_train, y_test), split_evidence = (
            split_acquirer.split(
                X,
                y,
                test_size=20,
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

        # 4.2 Instrument and execute pipeline.

        pipeline = run.instrument(pipeline)

        print("\nFitting Logistic Regression...")

        pipeline.fit(X_train, y_train)

        # 4.3 Evaluate on training and test datasets.

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

        # 4.5 Run diagnosis.

        result = run.result()

    # ------------------------------------------------------------
    # 5. Persist diagnostic run.
    # ------------------------------------------------------------

    saved_run_id = save_run(
        result,
        runtime_context=run.context,
        run_type="evaluation_uncertainty",
    )

    print(f"\nDashboard Run ID: {saved_run_id}")

    # ------------------------------------------------------------
    # 6. Calculate a reference Wilson interval.
    #
    # This is displayed for experiment interpretation.
    # PipeDiag independently calculates its G6 evidence.
    # ------------------------------------------------------------

    n = len(y_test)
    correct = int(
        np.sum(test_predictions == y_test)
    )

    p = correct / n
    z = 1.96

    denominator = 1 + z**2 / n

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )

    lower_bound = (
        p + z**2 / (2 * n) - margin
    ) / denominator

    upper_bound = (
        p + z**2 / (2 * n) + margin
    ) / denominator

    print("\n" + "=" * 70)
    print("REFERENCE EVALUATION UNCERTAINTY")
    print("=" * 70)

    print(f"Correct predictions: {correct}/{n}")
    print(f"Observed accuracy: {p:.4f}")
    print(f"Wilson lower bound: {lower_bound:.4f}")
    print(f"Wilson upper bound: {upper_bound:.4f}")
    print(f"Wilson interval half-width: {margin:.4f}")

    # ------------------------------------------------------------
    # 7. Display model evaluation.
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
            print(f"  {name}: {value}")

    # ------------------------------------------------------------
    # 10. Display deterministic diagnostic report.
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
    # 11. Display hypothesis classification.
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
        "artifacts/diagnostic_experiments/G6"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "run_id": run.context.run_id,
        "experiment": "intentional_evaluation_uncertainty",
        "expected_evidence": "G6",
        "dataset": "synthetic_small_evaluation_classification",
        "model": "LogisticRegression",
        "samples": len(X),
        "features": X.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "correct_predictions": correct,
            "evaluation_sample_count": n,
            "wilson_lower_bound": float(lower_bound),
            "wilson_upper_bound": float(upper_bound),
            "wilson_interval_half_width": float(margin),
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
    print("EVALUATION UNCERTAINTY EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
