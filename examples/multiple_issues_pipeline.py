
import json
import sys
from pathlib import Path
from dataclasses import replace

# Add project root so examples can import the experiments package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pipediag

from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
from pipediag.diagnostics.explanation import HYPOTHESIS_TITLES
from pipediag.evidence.runtime_provider import RuntimeEvidenceProvider
from pipediag.persistence import save_run

from experiments.multi_fault_validation import build_multi_fault_engine
from experiments.distribution_shift import DistributionShiftPipelineResult
from experiments.distribution_shift_evidence import (
    collect_distribution_shift_evidence,
)


# =============================================================
# DISTRIBUTION SHIFT EVIDENCE
# =============================================================

def build_actual_distribution_shift_evidence(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Calculate distribution shift using the actual experiment split.

    NaN values are excluded from individual feature statistics.

    The shift score is the mean absolute standardized difference
    between training and test feature means.
    """

    feature_shift_scores = []

    for feature_index in range(X_train.shape[1]):
        train_values = X_train[:, feature_index]
        test_values = X_test[:, feature_index]

        train_values = train_values[np.isfinite(train_values)]
        test_values = test_values[np.isfinite(test_values)]

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        train_std = float(np.std(train_values))

        if train_std <= 1e-12:
            continue

        train_mean = float(np.mean(train_values))
        test_mean = float(np.mean(test_values))

        shift_score = abs(test_mean - train_mean) / train_std

        feature_shift_scores.append(shift_score)

    shift_magnitude = (
        float(np.mean(feature_shift_scores))
        if feature_shift_scores
        else 0.0
    )

    shift_result = DistributionShiftPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        shift_magnitude=shift_magnitude,
        random_state=42,
    )

    return collect_distribution_shift_evidence(shift_result)


# =============================================================
# MAIN EXPERIMENT
# =============================================================

def main():

    print("=" * 70)
    print("MULTIPLE ISSUES - COMBINED DIAGNOSTIC EXPERIMENT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Generate an imbalanced classification dataset
    # ---------------------------------------------------------

    X, y = make_classification(
        n_samples=1000,
        n_features=25,
        n_informative=5,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.92, 0.08],
        flip_y=0.08,
        class_sep=0.7,
        random_state=42,
    )

    print(f"Original samples: {X.shape[0]}")
    print(f"Original features: {X.shape[1]}")

    # ---------------------------------------------------------
    # 2. Introduce constant features (G7)
    # ---------------------------------------------------------

    constant_features = np.column_stack(
        [
            np.zeros(X.shape[0]),
            np.ones(X.shape[0]),
            np.full(X.shape[0], 5.0),
            np.full(X.shape[0], -1.0),
            np.full(X.shape[0], 100.0),
        ]
    )

    X = np.column_stack([X, constant_features])

    # ---------------------------------------------------------
    # 3. Introduce missing values (G5)
    # ---------------------------------------------------------

    rng = np.random.default_rng(42)

    missing_mask = rng.random((X.shape[0], 25)) < 0.12
    X[:, :25][missing_mask] = np.nan

    # ---------------------------------------------------------
    # 4. Introduce distribution shift (G3)
    # ---------------------------------------------------------

    shift_start = int(X.shape[0] * 0.80)

    # Shift the first ten original features in the final 20%.
    X[shift_start:, :10] += 1.5

    print("\n" + "=" * 70)
    print("INTENTIONALLY INTRODUCED ISSUES")
    print("=" * 70)

    print("G3: Distribution shift in the final 20% of samples")
    print("G4: Imbalanced classes")
    print("G5: Approximately 12% missing values in original features")
    print("G6: Small test evaluation sample")
    print("G7: Five constant features")
    print("G2: Overfitting risk from an unrestricted decision tree")

    # ---------------------------------------------------------
    # 5. Build the sklearn pipeline
    # ---------------------------------------------------------

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------
    # 6. Configure PipeDiag
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
    # 7. Execute instrumented experiment
    # ---------------------------------------------------------

    with pipediag.start(
        application=diagnostic_application,
        metadata={
            "experiment": "multiple_issues_combined",
            "dataset": "synthetic_multi_issue_classification",
            "model": "DecisionTreeClassifier",
            "expected_evidence": [
                "G2",
                "G3",
                "G4",
                "G5",
                "G6",
                "G7",
            ],
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
                test_size=0.05,
                shuffle=False,
            )
        )

        print("\n" + "=" * 70)
        print("DATASET SPLIT")
        print("=" * 70)

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Testing class distribution: {np.bincount(y_test)}")

        # -----------------------------------------------------
        # 8. Instrument and train pipeline
        # -----------------------------------------------------

        pipeline = run.instrument(pipeline)

        print("\n" + "=" * 70)
        print("PIPELINE TRAINING")
        print("=" * 70)

        pipeline.fit(X_train, y_train)

        # -----------------------------------------------------
        # 9. Evaluate training and test data
        # -----------------------------------------------------

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)

        train_accuracy = float(
            np.mean(train_predictions == y_train)
        )

        test_accuracy = float(
            np.mean(test_predictions == y_test)
        )

        generalization_gap = train_accuracy - test_accuracy

        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy:  {test_accuracy:.4f}")
        print(f"Generalization gap: {generalization_gap:.4f}")

        # Create training evaluation evidence.
        train_evidence = EvaluationAcquirer().acquire_score(
            metric_name="accuracy",
            metric_value=train_accuracy,
            sample_count=len(y_train),
            evaluation_context=TRAIN,
        )

        # Create test evaluation evidence.
        test_evidence = EvaluationAcquirer().acquire_score(
            metric_name="accuracy",
            metric_value=test_accuracy,
            sample_count=len(y_test),
            evaluation_context=TEST,
        )

        run.context.add_evaluation(train_evidence)
        run.context.add_evaluation(test_evidence)

    # ---------------------------------------------------------
    # 10. Generate standard PipeDiag diagnostic result
    # ---------------------------------------------------------

        result = run.result()

    diagnostic = result.diagnostic_result

    # ---------------------------------------------------------
    # 11. Build actual G3 evidence
    # ---------------------------------------------------------

    actual_g3_evidence = build_actual_distribution_shift_evidence(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Replace unavailable G3 evidence from the runtime provider.
    multi_fault_evidence = [
        evidence
        for evidence in diagnostic.evidence
        if evidence.name != "G3"
    ]

    multi_fault_evidence.append(actual_g3_evidence)

    # ---------------------------------------------------------
    # 12. Run independent multi-fault inference
    # ---------------------------------------------------------

    multi_fault_engine = build_multi_fault_engine(
        detection_threshold=0.5
    )

    multi_fault_result = multi_fault_engine.diagnose(
        evidence_groups=multi_fault_evidence,
        metadata={
            "experiment": "multiple_issues_combined",
            "faults_injected": [
                "H2",
                "H3",
                "H4",
                "H5",
                "H6",
                "H8",
            ],
        },
    )




    # ---------------------------------------------------------
    # 13. Attach multi-fault result and persist the combined run
    # ---------------------------------------------------------

    # Persist the same evidence collection used for multi-fault inference.
    updated_diagnostic = replace(
        diagnostic,
        evidence=tuple(multi_fault_evidence),
    )

    result = replace(
        result,
        diagnostic_result=updated_diagnostic,
        multi_fault_result=multi_fault_result,
    )

    saved_run_id = save_run(
        result,
        runtime_context=run.context,
        run_type="multiple_issues",
    )

    # Keep the original single-fault diagnostic available.
    diagnostic = result.diagnostic_result

    # ---------------------------------------------------------
    # 14. Print diagnostic evidence
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DIAGNOSTIC EVIDENCE")
    print("=" * 70)

    evidence_results = []

    for evidence in multi_fault_evidence:

        print(f"\nEvidence Group: {evidence.name}")
        print(f"Status: {evidence.status}")
        print(f"Value: {evidence.value}")

        measurements = evidence.metadata.get("measurements", {})

        for name, value in measurements.items():
            print(f"  {name}: {value}")

        evidence_results.append(
            {
                "group": evidence.name,
                "status": str(evidence.status),
                "value": evidence.value,
                "measurements": measurements,
            }
        )

    # ---------------------------------------------------------
    # 15. Print standard single-fault posterior for comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STANDARD SINGLE-FAULT POSTERIOR (REFERENCE)")
    print("=" * 70)

    hypothesis_results = []

    supported_hypotheses = (
        diagnostic.evidence_supported_hypotheses or frozenset()
    )

    for hypothesis_id, posterior in sorted(
        diagnostic.posterior.items()
    ):

        title = HYPOTHESIS_TITLES.get(
            hypothesis_id,
            "Unknown Hypothesis",
        )

        evidence_supported = (
            hypothesis_id in supported_hypotheses
        )

        threshold_crossed = (
            posterior >= result.explanation.threshold
        )

        print(
            f"{hypothesis_id} | "
            f"{title} | "
            f"Posterior: {posterior:.4f} | "
            f"Evidence supported: {evidence_supported} | "
            f"Threshold crossed: {threshold_crossed}"
        )

        hypothesis_results.append(
            {
                "hypothesis_id": hypothesis_id,
                "title": title,
                "posterior": float(posterior),
                "evidence_supported": evidence_supported,
                "threshold_crossed": threshold_crossed,
            }
        )

    posterior_sum = sum(diagnostic.posterior.values())

    print("\n" + "-" * 70)
    print(f"STANDARD POSTERIOR SUM: {posterior_sum:.10f}")
    print(
        f"STANDARD REPORTING THRESHOLD: "
        f"{result.explanation.threshold:.4f}"
    )
    print("-" * 70)

    # ---------------------------------------------------------
    # 16. Print standard PipeDiag diagnostic report
    # ---------------------------------------------------------

    report = result.report

    print("\n" + "=" * 70)
    print("STANDARD PIPEDIAG DIAGNOSTIC REPORT")
    print("=" * 70)

    print(report)

    # ---------------------------------------------------------
    # 17. Prepare and print multi-fault results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("INDEPENDENT MULTI-FAULT DIAGNOSIS")
    print("=" * 70)

    multi_fault_hypothesis_results = []

    for hypothesis_id, posterior in sorted(
        multi_fault_result.posterior.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        title = HYPOTHESIS_TITLES.get(
            hypothesis_id,
            "Unknown Hypothesis",
        )

        detected = (
            hypothesis_id in multi_fault_result.detected_hypotheses
        )

        independent_probability = (
            multi_fault_result.independent_posterior[hypothesis_id]
        )

        print(
            f"{hypothesis_id} | "
            f"{title} | "
            f"Normalized support: {posterior:.4f} | "
            f"Independent probability: {independent_probability:.4f} | "
            f"Detected: {detected}"
        )

        multi_fault_hypothesis_results.append(
            {
                "hypothesis_id": hypothesis_id,
                "title": title,
                "normalized_support": float(posterior),
                "independent_probability": float(
                    independent_probability
                ),
                "detected": detected,
            }
        )

    multi_fault_posterior_sum = sum(
        multi_fault_result.posterior.values()
    )

    print("\nDetected simultaneous faults:")
    print(sorted(multi_fault_result.detected_hypotheses))

    print(
        "Normalized posterior sum:",
        round(multi_fault_posterior_sum, 10),
    )

    print(
        "Mean uncertainty:",
        round(multi_fault_result.uncertainty, 6),
    )

    print(
        "Detection threshold:",
        multi_fault_result.detection_threshold,
    )

    # ---------------------------------------------------------
    # 18. Save experiment results
    # ---------------------------------------------------------

    output_dir = Path(
        "artifacts/diagnostic_experiments/MULTIPLE_ISSUES"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "run_id": saved_run_id,
        "experiment": "multiple_issues_combined",

        "dataset": {
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "training_samples": int(len(X_train)),
            "testing_samples": int(len(X_test)),
            "training_accuracy": train_accuracy,
            "testing_accuracy": test_accuracy,
            "generalization_gap": generalization_gap,
            "training_class_distribution": np.bincount(
                y_train
            ).tolist(),
            "testing_class_distribution": np.bincount(
                y_test
            ).tolist(),
        },

        "intentionally_introduced_issues": [
            "distribution_shift",
            "class_imbalance",
            "missing_values",
            "small_test_sample",
            "constant_features",
            "overfitting_risk",
        ],

        "evidence": evidence_results,

        "single_fault_reference": {
            "hypotheses": hypothesis_results,
            "posterior_sum": float(posterior_sum),
        },

        "multi_fault_diagnosis": {
            "hypotheses": multi_fault_hypothesis_results,
            "normalized_posterior_sum": float(
                multi_fault_posterior_sum
            ),
            "detected_hypotheses": sorted(
                multi_fault_result.detected_hypotheses
            ),
            "uncertainty": float(
                multi_fault_result.uncertainty
            ),
            "detection_threshold": float(
                multi_fault_result.detection_threshold
            ),
            "evidence_log_likelihood_ratios": (
                multi_fault_result.evidence_log_likelihood_ratios
            ),
        },

        "report": report,
    }

    output_file = output_dir / "experiment_results.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            default=str,
        )

    # ---------------------------------------------------------
    # 19. Completion summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print(f"Dashboard Run ID: {saved_run_id}")

    print(
        "Detected faults:",
        sorted(multi_fault_result.detected_hypotheses),
    )

    print(
        f"Normalized multi-fault posterior sum: "
        f"{multi_fault_posterior_sum:.10f}"
    )

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
