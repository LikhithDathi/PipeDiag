
from __future__ import annotations

import numpy as np

from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import run_overfitting_pipeline


# -------------------------------------------------------------
# Experiment configuration
# -------------------------------------------------------------

SEED_POOL = tuple(range(1, 301))

N_PARTITIONS = 20
CALIBRATION_SIZE = 30
HELD_OUT_SIZE = 50

NORMAL_DEPTHS = (2, 3, 4, 5)

PARTITION_RANDOM_STATE = 2026
COVARIANCE_REGULARIZATION = 1e-4

FEATURES = (
    ("training_only", "A: Training accuracy only"),
    ("training_gap", "B: Training accuracy + generalization gap"),
    ("training_evaluation", "C: Training accuracy + evaluation accuracy"),
)


# -------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------

def get_features(
    seed: int,
    depth: int,
) -> dict[str, np.ndarray]:
    """Extract candidate H7 features for one seed and depth."""

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=seed,
        max_depth=depth,
    )

    evidence = collect_generalization_evidence(result)
    measurements = evidence.metadata["measurements"]

    train = float(measurements["training_accuracy"].value)
    evaluation = float(measurements["evaluation_accuracy"].value)
    gap = float(measurements["generalization_gap"].value)

    return {
        "training_only": np.array(
            [train],
            dtype=float,
        ),
        "training_gap": np.array(
            [train, gap],
            dtype=float,
        ),
        "training_evaluation": np.array(
            [train, evaluation],
            dtype=float,
        ),
    }


def build_feature_cache(
    seeds: set[int],
) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """Calculate each seed/depth combination only once."""

    cache: dict[
        int,
        dict[int, dict[str, np.ndarray]],
    ] = {}

    for index, seed in enumerate(sorted(seeds), start=1):
        cache[seed] = {
            depth: get_features(seed, depth)
            for depth in (1, *NORMAL_DEPTHS)
        }

        if index % 25 == 0 or index == len(seeds):
            print(
                f"  Cached {index}/{len(seeds)} seeds"
            )

    return cache


# -------------------------------------------------------------
# Partition generation
# -------------------------------------------------------------

def generate_partitions() -> list[
    tuple[tuple[int, ...], tuple[int, ...]]
]:
    """Generate reproducible, internally disjoint partitions."""

    required = CALIBRATION_SIZE + HELD_OUT_SIZE

    if required > len(SEED_POOL):
        raise ValueError(
            "Seed pool is too small for the requested partition"
        )

    rng = np.random.default_rng(
        PARTITION_RANDOM_STATE
    )

    partitions = []

    for _ in range(N_PARTITIONS):
        selected = rng.choice(
            SEED_POOL,
            size=required,
            replace=False,
        )

        calibration = tuple(
            int(seed)
            for seed in selected[:CALIBRATION_SIZE]
        )

        held_out = tuple(
            int(seed)
            for seed in selected[CALIBRATION_SIZE:]
        )

        if set(calibration) & set(held_out):
            raise RuntimeError(
                "Calibration and held-out seeds overlap"
            )

        partitions.append(
            (calibration, held_out)
        )

    return partitions


# -------------------------------------------------------------
# Gaussian likelihood model
# -------------------------------------------------------------

def fit_gaussian(
    samples: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian distribution using sample covariance."""

    matrix = np.asarray(samples, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(
            "Samples must form a two-dimensional matrix"
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two observations are required"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Samples must contain only finite values"
        )

    mean = matrix.mean(axis=0)

    covariance = np.atleast_2d(
        np.cov(matrix, rowvar=False, ddof=1)
    )

    covariance += (
        np.eye(matrix.shape[1])
        * COVARIANCE_REGULARIZATION
    )

    return mean, covariance


def log_density(
    x: np.ndarray,
    model: tuple[np.ndarray, np.ndarray],
) -> float:
    """Calculate multivariate Gaussian log density."""

    mean, covariance = model

    x = np.asarray(x, dtype=float)

    if x.shape != mean.shape:
        raise ValueError(
            "Observation dimensions do not match model dimensions"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Observation must contain only finite values"
        )

    delta = x - mean

    sign, logdet = np.linalg.slogdet(covariance)

    if sign <= 0:
        raise ValueError(
            "Covariance matrix is not positive definite"
        )

    dimension = len(mean)

    quadratic = delta @ np.linalg.solve(
        covariance,
        delta,
    )

    return float(
        -0.5
        * (
            dimension * np.log(2.0 * np.pi)
            + logdet
            + quadratic
        )
    )


def fit_binary_model(
    positive: list[np.ndarray],
    negative: list[np.ndarray],
):
    """Fit positive and negative Gaussian distributions."""

    if not positive or not negative:
        raise ValueError(
            "Both positive and negative samples are required"
        )

    return (
        fit_gaussian(positive),
        fit_gaussian(negative),
    )


def log_likelihood_ratio(
    x: np.ndarray,
    model,
) -> float:
    """Calculate log p(x|H7) - log p(x|not H7)."""

    positive_model, negative_model = model

    return (
        log_density(x, positive_model)
        - log_density(x, negative_model)
    )


# -------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------

def evaluate_partition(
    feature_name: str,
    feature_cache: dict[
        int,
        dict[int, dict[str, np.ndarray]],
    ],
    calibration_seeds: tuple[int, ...],
    held_out_seeds: tuple[int, ...],
) -> dict[str, float | int]:
    """Fit a fresh model and evaluate one partition."""

    if set(calibration_seeds) & set(held_out_seeds):
        raise ValueError(
            "Calibration and held-out seeds must be disjoint"
        )

    positive_train = []
    negative_train = []

    # Fit using calibration seeds only.
    for seed in calibration_seeds:
        positive_train.append(
            feature_cache[seed][1][feature_name]
        )

        for depth in NORMAL_DEPTHS:
            negative_train.append(
                feature_cache[seed][depth][feature_name]
            )

    model = fit_binary_model(
        positive_train,
        negative_train,
    )

    tp = fn = tn = fp = 0

    seed_positive_scores = []
    seed_negative_scores = []

    # Evaluate using held-out seeds only.
    for seed in held_out_seeds:
        positive_x = feature_cache[seed][1][feature_name]

        positive_correct = (
            log_likelihood_ratio(positive_x, model) > 0
        )

        if positive_correct:
            tp += 1
        else:
            fn += 1

        negative_correct = []

        for depth in NORMAL_DEPTHS:
            negative_x = feature_cache[seed][depth][feature_name]

            is_correct = (
                log_likelihood_ratio(negative_x, model) <= 0
            )

            negative_correct.append(is_correct)

            if is_correct:
                tn += 1
            else:
                fp += 1

        seed_positive_scores.append(
            float(positive_correct)
        )

        seed_negative_scores.append(
            float(np.mean(negative_correct))
        )

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    seed_sensitivity = float(
        np.mean(seed_positive_scores)
    )

    seed_specificity = float(
        np.mean(seed_negative_scores)
    )

    seed_balanced_accuracy = (
        seed_sensitivity + seed_specificity
    ) / 2.0

    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "seed_sensitivity": seed_sensitivity,
        "seed_specificity": seed_specificity,
        "seed_balanced_accuracy": seed_balanced_accuracy,
    }


# -------------------------------------------------------------
# Summary statistics
# -------------------------------------------------------------

def summarize_results(
    values: list[float],
) -> dict[str, float]:
    """Calculate descriptive statistics across partitions."""

    array = np.asarray(values, dtype=float)

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def summarize_paired_differences(
    values_b: list[float],
    values_a: list[float],
) -> dict[str, float]:
    """Summarize paired differences between two configurations."""

    if len(values_a) != len(values_b):
        raise ValueError(
            "Paired result lists must have equal lengths"
        )

    differences = (
        np.asarray(values_b, dtype=float)
        - np.asarray(values_a, dtype=float)
    )

    return {
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(
            np.std(differences, ddof=1)
        ),
        "minimum_difference": float(np.min(differences)),
        "maximum_difference": float(np.max(differences)),
        "fraction_b_better": float(
            np.mean(differences > 0)
        ),
    }


# -------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("H7 REPEATED CALIBRATION PARTITION EXPERIMENT")
    print("=" * 65)

    print(f"Seed pool size: {len(SEED_POOL)}")
    print(f"Repeated partitions: {N_PARTITIONS}")
    print(f"Calibration seeds per partition: {CALIBRATION_SIZE}")
    print(f"Held-out seeds per partition: {HELD_OUT_SIZE}")
    print(f"Normal depths: {NORMAL_DEPTHS}")

    # ---------------------------------------------------------
    # Generate partitions
    # ---------------------------------------------------------

    partitions = generate_partitions()

    all_seeds = set()

    for calibration, held_out in partitions:
        all_seeds.update(calibration)
        all_seeds.update(held_out)

    print("\nGenerating feature cache...")
    print(f"Unique seeds required: {len(all_seeds)}")

    feature_cache = build_feature_cache(
        all_seeds
    )

    print("\nFeature cache completed.")

    # ---------------------------------------------------------
    # Run repeated experiments
    # ---------------------------------------------------------

    partition_results = {
        feature_name: []
        for feature_name, _ in FEATURES
    }

    for partition_index, (
        calibration_seeds,
        held_out_seeds,
    ) in enumerate(partitions, start=1):

        print(
            f"\nPartition {partition_index}/{N_PARTITIONS}"
        )

        print(
            f"  Calibration seeds: {len(calibration_seeds)}"
        )

        print(
            f"  Held-out seeds: {len(held_out_seeds)}"
        )

        for feature_name, label in FEATURES:
            result = evaluate_partition(
                feature_name,
                feature_cache,
                calibration_seeds,
                held_out_seeds,
            )

            partition_results[feature_name].append(
                result
            )

            print(
                f"  {label}: "
                f"{result['seed_balanced_accuracy']:.1%}"
            )

    # ---------------------------------------------------------
    # Overall feature comparison
    # ---------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("REPEATED PARTITION SUMMARY")
    print("=" * 65)

    summary_results = {}

    for feature_name, label in FEATURES:
        scores = [
            result["seed_balanced_accuracy"]
            for result in partition_results[feature_name]
        ]

        summary = summarize_results(scores)

        summary_results[feature_name] = summary

        print(f"\n{label}")

        print(
            f"  Mean balanced accuracy: "
            f"{summary['mean']:.2%}"
        )

        print(
            f"  Standard deviation:     "
            f"{summary['std']:.2%}"
        )

        print(
            f"  Minimum:                "
            f"{summary['minimum']:.2%}"
        )

        print(
            f"  Maximum:                "
            f"{summary['maximum']:.2%}"
        )

    # ---------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("RANKING BY MEAN BALANCED ACCURACY")
    print("=" * 65)

    ranking = sorted(
        FEATURES,
        key=lambda item: summary_results[item[0]]["mean"],
        reverse=True,
    )

    for rank, (feature_name, label) in enumerate(
        ranking,
        start=1,
    ):
        summary = summary_results[feature_name]

        print(
            f"{rank}. {label}: "
            f"{summary['mean']:.2%} "
            f"(SD={summary['std']:.2%})"
        )

    # ---------------------------------------------------------
    # Paired comparisons
    # ---------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("PAIRED PARTITION COMPARISONS")
    print("=" * 65)

    comparisons = (
        (
            "B vs A",
            "training_gap",
            "training_only",
        ),
        (
            "B vs C",
            "training_gap",
            "training_evaluation",
        ),
    )

    for comparison, feature_b, feature_a in comparisons:
        scores_b = [
            result["seed_balanced_accuracy"]
            for result in partition_results[feature_b]
        ]

        scores_a = [
            result["seed_balanced_accuracy"]
            for result in partition_results[feature_a]
        ]

        summary = summarize_paired_differences(
            scores_b,
            scores_a,
        )

        print(f"\n{comparison}")

        print(
            f"  Mean difference: "
            f"{summary['mean_difference']:+.2%}"
        )

        print(
            f"  Difference SD: "
            f"{summary['std_difference']:.2%}"
        )

        print(
            f"  Minimum difference: "
            f"{summary['minimum_difference']:+.2%}"
        )

        print(
            f"  Maximum difference: "
            f"{summary['maximum_difference']:+.2%}"
        )

        print(
            f"  Fraction of partitions where B is better: "
            f"{summary['fraction_b_better']:.1%}"
        )

    # ---------------------------------------------------------
    # Methodological note
    # ---------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("EXPERIMENT NOTES")
    print("=" * 65)

    print(
        "1. Gaussian models are refitted for every partition."
    )

    print(
        "2. Calibration and held-out seeds are disjoint "
        "within each partition."
    )

    print(
        "3. Seeds may recur across different partitions."
    )

    print(
        "4. Partition results are therefore descriptive "
        "stability measurements, not independent replications."
    )

    print(
        "5. No production likelihood or registry code is modified."
    )


if __name__ == "__main__":
    main()
