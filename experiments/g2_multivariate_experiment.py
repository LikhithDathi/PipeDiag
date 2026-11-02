
from __future__ import annotations

import numpy as np

from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import run_overfitting_pipeline


CALIBRATION_SEEDS = tuple(range(1, 31))
HELD_OUT_SEEDS = tuple(range(200, 250))
NORMAL_DEPTHS = (2, 3, 4, 5)

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_RANDOM_STATE = 42
COVARIANCE_REGULARIZATION = 1e-4


def get_features(
    seed: int,
    depth: int | None,
) -> dict[str, np.ndarray]:
    """Extract the three candidate H7 feature representations."""
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
        "training_only": np.array([train], dtype=float),
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
    seeds: tuple[int, ...],
) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """Calculate each seed/depth combination only once."""
    cache: dict[int, dict[int, dict[str, np.ndarray]]] = {}

    for seed in seeds:
        cache[seed] = {
            depth: get_features(seed, depth)
            for depth in (1, *NORMAL_DEPTHS)
        }

    return cache


def fit_gaussian(
    samples: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a multivariate Gaussian using sample covariance."""
    matrix = np.asarray(samples, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(
            "Samples must form a two-dimensional matrix"
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two samples are required"
        )

    if matrix.shape[1] < 1:
        raise ValueError(
            "At least one feature is required"
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
    """Calculate the log probability density of a Gaussian model."""
    mean, covariance = model

    x = np.asarray(x, dtype=float)

    if x.shape != mean.shape:
        raise ValueError(
            "Observation dimensions must match model dimensions"
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
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Fit positive and negative class Gaussian distributions."""
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
    model: tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ],
) -> float:
    """Calculate log p(x|H7) - log p(x|not H7)."""
    positive_model, negative_model = model

    return (
        log_density(x, positive_model)
        - log_density(x, negative_model)
    )


def evaluate_h7_feature_set(
    feature_name: str,
    feature_cache: dict[
        int,
        dict[int, dict[str, np.ndarray]],
    ],
    calibration_seeds: tuple[int, ...] = CALIBRATION_SEEDS,
    held_out_seeds: tuple[int, ...] = HELD_OUT_SEEDS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float | int | list[float]]:
    """Fit and evaluate one H7 feature representation."""
    if not calibration_seeds:
        raise ValueError(
            "Calibration seeds cannot be empty"
        )

    if not held_out_seeds:
        raise ValueError(
            "Held-out seeds cannot be empty"
        )

    if len(set(calibration_seeds)) != len(calibration_seeds):
        raise ValueError(
            "Calibration seeds must be unique"
        )

    if len(set(held_out_seeds)) != len(held_out_seeds):
        raise ValueError(
            "Held-out seeds must be unique"
        )

    if set(calibration_seeds) & set(held_out_seeds):
        raise ValueError(
            "Calibration and held-out seeds must be disjoint"
        )

    if bootstrap_samples < 1:
        raise ValueError(
            "bootstrap_samples must be at least 1"
        )

    required_seeds = (
        set(calibration_seeds)
        | set(held_out_seeds)
    )

    missing_seeds = required_seeds - set(feature_cache)

    if missing_seeds:
        raise ValueError(
            f"Feature cache is missing seeds: {sorted(missing_seeds)}"
        )

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------

    positive_train = []
    negative_train = []

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

    # ---------------------------------------------------------
    # Held-out evaluation
    # ---------------------------------------------------------

    tp = fn = tn = fp = 0

    seed_positive_correct: list[float] = []
    seed_negative_correct_rates: list[float] = []
    seed_scores: list[float] = []

    for seed in held_out_seeds:
        positive_x = feature_cache[seed][1][feature_name]

        positive_lr = log_likelihood_ratio(
            positive_x,
            model,
        )

        positive_correct = positive_lr > 0

        if positive_correct:
            tp += 1
        else:
            fn += 1

        negative_correct: list[bool] = []

        for depth in NORMAL_DEPTHS:
            negative_x = feature_cache[seed][depth][feature_name]

            negative_lr = log_likelihood_ratio(
                negative_x,
                model,
            )

            is_correct = negative_lr <= 0
            negative_correct.append(is_correct)

            if is_correct:
                tn += 1
            else:
                fp += 1

        seed_positive_correct.append(
            float(positive_correct)
        )

        seed_negative_correct_rates.append(
            float(np.mean(negative_correct))
        )

        # Each seed contributes equally to the final score.
        seed_scores.append(
            (
                float(positive_correct)
                + float(np.mean(negative_correct))
            ) / 2.0
        )

    # ---------------------------------------------------------
    # Configuration-level metrics
    # ---------------------------------------------------------

    sensitivity = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    # ---------------------------------------------------------
    # Seed-balanced metrics
    # ---------------------------------------------------------

    seed_sensitivity = float(
        np.mean(seed_positive_correct)
    )

    seed_specificity = float(
        np.mean(seed_negative_correct_rates)
    )

    seed_balanced_accuracy = (
        seed_sensitivity + seed_specificity
    ) / 2.0

    # ---------------------------------------------------------
    # Seed-level bootstrap confidence interval
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    n_seeds = len(held_out_seeds)

    positive_array = np.asarray(
        seed_positive_correct,
        dtype=float,
    )

    negative_array = np.asarray(
        seed_negative_correct_rates,
        dtype=float,
    )

    bootstrap_scores = np.empty(
        bootstrap_samples,
        dtype=float,
    )

    for index in range(bootstrap_samples):
        indices = rng.integers(
            0,
            n_seeds,
            size=n_seeds,
        )

        sampled_sensitivity = float(
            np.mean(positive_array[indices])
        )

        sampled_specificity = float(
            np.mean(negative_array[indices])
        )

        bootstrap_scores[index] = (
            sampled_sensitivity
            + sampled_specificity
        ) / 2.0

    ci_lower, ci_upper = np.percentile(
        bootstrap_scores,
        [2.5, 97.5],
    )

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
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "seed_scores": seed_scores,
    }


def paired_bootstrap_comparison(
    scores_a: list[float],
    scores_b: list[float],
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float]:
    """Compare two feature sets using paired seed-level bootstrap."""
    if len(scores_a) != len(scores_b):
        raise ValueError(
            "Paired score arrays must have equal lengths"
        )

    if len(scores_a) < 2:
        raise ValueError(
            "At least two paired seeds are required"
        )

    if bootstrap_samples < 1:
        raise ValueError(
            "bootstrap_samples must be at least 1"
        )

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if not np.all(np.isfinite(a)):
        raise ValueError(
            "scores_a must contain only finite values"
        )

    if not np.all(np.isfinite(b)):
        raise ValueError(
            "scores_b must contain only finite values"
        )

    if np.any((a < 0) | (a > 1)):
        raise ValueError(
            "scores_a must contain values between 0 and 1"
        )

    if np.any((b < 0) | (b > 1)):
        raise ValueError(
            "scores_b must contain values between 0 and 1"
        )

    # Positive differences mean B performs better than A.
    differences = b - a

    observed_difference = float(
        np.mean(differences)
    )

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    n = len(differences)

    bootstrap_differences = np.empty(
        bootstrap_samples,
        dtype=float,
    )

    for index in range(bootstrap_samples):
        sampled_indices = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_differences[index] = np.mean(
            differences[sampled_indices]
        )

    lower, upper = np.percentile(
        bootstrap_differences,
        [2.5, 97.5],
    )

    return {
        "observed_difference": observed_difference,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "probability_b_better": float(
            np.mean(bootstrap_differences > 0)
        ),
    }


def main() -> None:
    candidates = (
        (
            "training_only",
            "A: Training accuracy only",
        ),
        (
            "training_gap",
            "B: Training accuracy + generalization gap",
        ),
        (
            "training_evaluation",
            "C: Training accuracy + evaluation accuracy",
        ),
    )

    all_seeds = tuple(
        sorted(
            set(CALIBRATION_SEEDS)
            | set(HELD_OUT_SEEDS)
        )
    )

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------

    print("Building feature cache...")

    feature_cache = build_feature_cache(
        all_seeds
    )

    print(
        f"Cached {len(all_seeds)} seeds."
    )

    print("\nH7 FEATURE COMPARISON")

    print(
        f"Calibration seeds: "
        f"{CALIBRATION_SEEDS[0]}-{CALIBRATION_SEEDS[-1]}"
    )

    print(
        f"Held-out seeds: "
        f"{HELD_OUT_SEEDS[0]}-{HELD_OUT_SEEDS[-1]}"
    )

    print(
        "Positive cases:",
        len(HELD_OUT_SEEDS),
    )

    print(
        "Normal comparisons:",
        len(HELD_OUT_SEEDS) * len(NORMAL_DEPTHS),
    )

    print(
        "\nA positive log-likelihood ratio supports H7."
    )

    print(
        "A negative or zero ratio does not support H7."
    )

    # ---------------------------------------------------------
    # Evaluate candidate feature representations
    # ---------------------------------------------------------

    results = []

    for feature_name, label in candidates:
        result = evaluate_h7_feature_set(
            feature_name,
            feature_cache,
        )

        results.append(
            (label, result)
        )

        print(f"\n{label}")

        print(
            f"  TP={result['tp']}, "
            f"FN={result['fn']}, "
            f"TN={result['tn']}, "
            f"FP={result['fp']}"
        )

        print(
            f"  Configuration sensitivity:       "
            f"{result['sensitivity']:.1%}"
        )

        print(
            f"  Configuration specificity:       "
            f"{result['specificity']:.1%}"
        )

        print(
            f"  Configuration balanced accuracy: "
            f"{result['balanced_accuracy']:.1%}"
        )

        print(
            f"  Seed-balanced sensitivity:       "
            f"{result['seed_sensitivity']:.1%}"
        )

        print(
            f"  Seed-balanced specificity:       "
            f"{result['seed_specificity']:.1%}"
        )

        print(
            f"  Seed-balanced accuracy:          "
            f"{result['seed_balanced_accuracy']:.1%}"
        )

        # This is an individual seed-bootstrap interval,
        # not a paired comparison interval.
        print(
            "  Seed-bootstrap 95% CI:           "
            f"[{result['ci_lower']:.3%}, "
            f"{result['ci_upper']:.3%}]"
        )

    # ---------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------

    print(
        "\nRANKING BY SEED-BALANCED ACCURACY"
    )

    for label, result in sorted(
        results,
        key=lambda item: (
            item[1]["seed_balanced_accuracy"],
            item[1]["seed_sensitivity"],
            item[1]["seed_specificity"],
        ),
        reverse=True,
    ):
        print(
            f"  {label}: "
            f"{result['seed_balanced_accuracy']:.1%} "
            f"(95% CI: "
            f"{result['ci_lower']:.1%}-"
            f"{result['ci_upper']:.1%})"
        )

    # ---------------------------------------------------------
    # Paired bootstrap comparisons
    # ---------------------------------------------------------

    result_map = {
        label: result
        for label, result in results
    }

    comparisons = (
        (
            "B vs A",
            "B: Training accuracy + generalization gap",
            "A: Training accuracy only",
        ),
        (
            "B vs C",
            "B: Training accuracy + generalization gap",
            "C: Training accuracy + evaluation accuracy",
        ),
    )

    print(
        "\nPAIRED BOOTSTRAP COMPARISONS"
    )

    for comparison, label_b, label_a in comparisons:
        result = paired_bootstrap_comparison(
            result_map[label_a]["seed_scores"],
            result_map[label_b]["seed_scores"],
        )

        print(f"\n{comparison}")

        print(
            "  Balanced accuracy difference: "
            f"{result['observed_difference']:+.3%}"
        )

        print(
            "  Paired bootstrap 95% CI: "
            f"[{result['ci_lower']:+.3%}, "
            f"{result['ci_upper']:+.3%}]"
        )

        print(
            "  Bootstrap fraction with B better: "
            f"{result['probability_b_better']:.1%}"
        )


if __name__ == "__main__":
    main()
