
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pipediag

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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


SCENARIOS = [
    # Experiment A: Vary H3, keep H2 fixed.
    {
        "name": "shift_mild",
        "shift_strength": 0.25,
        "tree_depth": None,
    },
    {
        "name": "shift_moderate",
        "shift_strength": 0.60,
        "tree_depth": None,
    },
    {
        "name": "shift_strong",
        "shift_strength": 1.00,
        "tree_depth": None,
    },

    # Experiment B: Vary H2, keep H3 fixed.
    {
        "name": "overfit_mild",
        "shift_strength": 0.60,
        "tree_depth": 4,
    },
    {
        "name": "overfit_moderate",
        "shift_strength": 0.60,
        "tree_depth": 8,
    },
    {
        "name": "overfit_strong",
        "shift_strength": 0.60,
        "tree_depth": None,
    },
]

RANDOM_SEEDS = [42, 123, 2026]


def calculate_g3_evidence(
    X_train, X_test, y_train, y_test, random_state=42
):
    scores = []

    for index in range(X_train.shape[1]):
        train = X_train[:, index]
        test = X_test[:, index]

        train = train[np.isfinite(train)]
        test = test[np.isfinite(test)]

        if len(train) == 0 or len(test) == 0:
            continue

        std = float(np.std(train))

        if std <= 1e-12:
            continue

        score = abs(float(np.mean(test)) - float(np.mean(train))) / std
        scores.append(score)

    shift = float(np.mean(scores)) if scores else 0.0

    shift_result = DistributionShiftPipelineResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        shift_magnitude=shift,
        random_state=random_state,
    )

    return collect_distribution_shift_evidence(shift_result)


def run_scenario(scenario, random_state):
    name = scenario["name"]
    shift_strength = scenario["shift_strength"]
    tree_depth = scenario["tree_depth"]

    print("\n" + "=" * 75)
    print(f"SCENARIO: {name.upper()}")
    print("=" * 75)

    # Generate a balanced dataset with informative and noise features.
    X, y = make_classification(
        n_samples=1500,
        n_features=30,
        n_informative=6,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.5, 0.5],
        flip_y=0.02,
        class_sep=0.8,
        random_state=random_state,
    )

    X = X.astype(float)

    # Inject H3: shift the final 20% of samples.
    split_index = int(len(X) * 0.80)
    X[split_index:, :5] += shift_strength

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=tree_depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    application = DiagnosticApplication(
        evidence_provider=DiagnosticEvidenceProvider(
            RuntimeEvidenceProvider()
        ),
        diagnostic_engine=build_calibrated_diagnostic_engine(),
    )

    with pipediag.start(
        application=application,
        metadata={
            "experiment": "two_fault_severity",
            "scenario": name,
            "injected_faults": ["H2", "H3"],
            "shift_strength": shift_strength,
            "tree_depth": tree_depth,
            "random_seed": random_state,
        },
    ) as run:

        split_acquirer = SplitAcquirer(
            lineage=run.context.evidence_collector.lineage,
            registry=run.context.evidence_collector.registry,
            context_registry=run.context.dataset_contexts,
        )

        (X_train, X_test, y_train, y_test), _ = (
            split_acquirer.split(
                X,
                y,
                test_size=0.20,
                shuffle=False,
            )
        )

        pipeline = run.instrument(pipeline)
        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)

        train_accuracy = float(
            np.mean(train_predictions == y_train)
        )
        test_accuracy = float(
            np.mean(test_predictions == y_test)
        )
        gap = train_accuracy - test_accuracy

        run.context.add_evaluation(
            EvaluationAcquirer().acquire_score(
                metric_name="accuracy",
                metric_value=train_accuracy,
                sample_count=len(y_train),
                evaluation_context=TRAIN,
            )
        )

        run.context.add_evaluation(
            EvaluationAcquirer().acquire_score(
                metric_name="accuracy",
                metric_value=test_accuracy,
                sample_count=len(y_test),
                evaluation_context=TEST,
            )
        )

        result = run.result()

    diagnostic = result.diagnostic_result

    # Replace runtime G3 with the shift measurement from this
    # experiment's actual train/test feature arrays.
    g3 = calculate_g3_evidence(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=random_state,
    )

    evidence = [
        item for item in diagnostic.evidence
        if item.name != "G3"
    ]
    evidence.append(g3)

    multi_fault_engine = build_multi_fault_engine(
        detection_threshold=0.5
    )

    multi_fault_result = multi_fault_engine.diagnose(
        evidence_groups=evidence,
        metadata={
            "experiment": "two_fault_severity",
            "scenario": name,
            "faults_injected": ["H2", "H3"],
            "shift_strength": shift_strength,
            "tree_depth": tree_depth,
            "random_seed": random_state,
        },
    )

    # Persist the same evidence used by multi-fault inference.
    updated_diagnostic = replace(
        diagnostic,
        evidence=tuple(evidence),
    )

    result = replace(
        result,
        diagnostic_result=updated_diagnostic,
        multi_fault_result=multi_fault_result,
    )

    run_id = save_run(
        result,
        runtime_context=run.context,
        run_type=f"two_fault_{name}",
    )

    # Print scenario measurements.
    print(f"Shift strength:       {shift_strength}")
    print(f"Decision tree depth:  {tree_depth}")
    print(f"Training accuracy:    {train_accuracy:.4f}")
    print(f"Testing accuracy:     {test_accuracy:.4f}")
    print(f"Generalization gap:   {gap:.4f}")
    print(f"G3 shift value:       {g3.value}")

    print("\nMULTI-FAULT PROBABILITIES")
    print("-" * 75)

    hypotheses = []

    for hypothesis_id in multi_fault_result.posterior:
        normalized = multi_fault_result.posterior[hypothesis_id]
        independent = multi_fault_result.independent_posterior[
            hypothesis_id
        ]

        detected = (
            hypothesis_id in multi_fault_result.detected_hypotheses
        )

        title = HYPOTHESIS_TITLES.get(
            hypothesis_id,
            "Unknown",
        )

        print(
            f"{hypothesis_id} | {title:28s} | "
            f"Independent: {independent:.6f} | "
            f"Normalized: {normalized:.6f} | "
            f"Detected: {detected}"
        )

        hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "title": title,
                "independent_probability": float(independent),
                "normalized_support": float(normalized),
                "detected": detected,
            }
        )

    print("\nDetected faults:")
    print(sorted(multi_fault_result.detected_hypotheses))

    print(f"Run ID: {run_id}")

    return {
        "scenario": name,
        "run_id": run_id,
        "random_seed": random_state,
        "injected_faults": ["H2", "H3"],
        "injection": {
            "shift_strength": shift_strength,
            "tree_depth": tree_depth,
        },
        "measurements": {
            "training_accuracy": train_accuracy,
            "testing_accuracy": test_accuracy,
            "generalization_gap": gap,
            "g3_shift": float(g3.value),
        },
        "hypotheses": hypotheses,
        "detected_hypotheses": sorted(
            multi_fault_result.detected_hypotheses
        ),
        "normalized_posterior_sum": float(
            sum(multi_fault_result.posterior.values())
        ),
        "uncertainty": float(multi_fault_result.uncertainty),
    }


def main():
    print("=" * 75)
    print("TWO-FAULT SEVERITY REPEATABILITY EXPERIMENT")
    print("H2: Generalization Failure")
    print("H3: Distribution Shift")
    print(f"Random seeds: {RANDOM_SEEDS}")
    print("=" * 75)

    results = []

    for seed in RANDOM_SEEDS:
        print("\n" + "#" * 75)
        print(f"RANDOM SEED: {seed}")
        print("#" * 75)

        for scenario in SCENARIOS:
            results.append(
                run_scenario(scenario, random_state=seed)
            )

    output_dir = Path(
        "artifacts/diagnostic_experiments/TWO_FAULT_SEVERITY"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual experiment results.
    output_file = output_dir / "multi_seed_severity_results.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Aggregate H2 and H3 probabilities across random seeds.
    print("\n" + "=" * 75)
    print("REPEATABILITY SUMMARY")
    print("=" * 75)

    for scenario in SCENARIOS:
        name = scenario["name"]

        scenario_results = [
            result for result in results
            if result["scenario"] == name
        ]

        print(f"\n{name.upper()}")

        for hypothesis_id in ("H2", "H3"):
            probabilities = []

            for result in scenario_results:
                hypothesis = next(
                    item for item in result["hypotheses"]
                    if item["hypothesis_id"] == hypothesis_id
                )

                probabilities.append(
                    hypothesis["independent_probability"]
                )

            print(
                f"{hypothesis_id}: "
                f"Mean={np.mean(probabilities):.6f}, "
                f"Std={np.std(probabilities):.6f}, "
                f"Min={np.min(probabilities):.6f}, "
                f"Max={np.max(probabilities):.6f}"
            )

        detected_sets = [
            set(result["detected_hypotheses"])
            for result in scenario_results
        ]

        consistent = all(
            detected == detected_sets[0]
            for detected in detected_sets
        )

        print(f"Detection consistent across seeds: {consistent}")
        print(f"Detected faults: {sorted(detected_sets[0])}")

    print("\n" + "=" * 75)
    print(f"Total experiment runs: {len(results)}")
    print(f"Results saved to: {output_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
