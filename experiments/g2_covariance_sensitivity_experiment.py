
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

REGULARIZATION_VALUES = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
)

# C = A @ B, where:
# B = [training_accuracy, generalization_gap]
# C = [training_accuracy, evaluation_accuracy]
#
# evaluation_accuracy = training_accuracy - generalization_gap

FEATURE_TRANSFORM = np.array(
    [
        [1.0, 0.0],
        [1.0, -1.0],
    ],
    dtype=float,
)


# -------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------

def get_features(
    seed: int,
    depth: int,
) -> dict[str, np.ndarray]:
    """Extract the H7 feature representations."""

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=seed,
        max_depth=depth,
    )

    evidence = collect_generalization_evidence(result)
    measurements = evidence.metadata["measurements"]

    train = float(
        measurements["training_accuracy"].value
    )

    evaluation = float(
        measurements["evaluation_accuracy"].value
    )

    gap = float(
        measurements["generalization_gap"].value
    )

    return {
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
    """Calculate every seed/depth combination once."""

    cache: dict[
        int,
        dict[int, dict[str, np.ndarray]],
    ] = {}

    ordered_seeds = sorted(seeds)

    for index, seed in enumerate(ordered_seeds, start=1):
        cache[seed] = {
            depth: get_features(seed, depth)
            for depth in (1, *NORMAL_DEPTHS)
        }

        if index % 25 == 0 or index == len(ordered_seeds):
            print(
                f"  Cached {index}/{len(ordered_seeds)} seeds"
            )

    return cache


# -------------------------------------------------------------
# Partition generation
# -------------------------------------------------------------

def generate_partitions() -> list[
    tuple[tuple[int, ...], tuple[int, ...]]
]:
    """Generate reproducible calibration/held-out partitions."""

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
# Gaussian model fitting
# -------------------------------------------------------------

def fit_gaussian(
    samples: list[np.ndarray],
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian using sample covariance and diagonal loading."""

    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError(
            "Regularization must be positive and finite"
        )

    matrix = np.asarray(
        samples,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Samples must form a two-dimensional matrix"
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two samples are required"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Samples must contain only finite values"
        )

    mean = matrix.mean(axis=0)

    covariance = np.atleast_2d(
        np.cov(
            matrix,
            rowvar=False,
            ddof=1,
        )
    )

    covariance += (
        np.eye(matrix.shape[1])
        * regularization
    )

    return mean, covariance


def fit_binary_model(
    positive: list[np.ndarray],
    negative: list[np.ndarray],
    regularization: float,
):
    """Fit positive and negative Gaussian distributions."""

    if not positive or not negative:
        raise ValueError(
            "Both positive and negative samples are required"
        )

    return (
        fit_gaussian(positive, regularization),
        fit_gaussian(negative, regularization),
    )


# -------------------------------------------------------------
# Gaussian likelihood calculations
# -------------------------------------------------------------

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
# Affine transformation of Gaussian models
# -------------------------------------------------------------

def transform_gaussian_model(
    model: tuple[np.ndarray, np.ndarray],
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a Gaussian mean and covariance into new coordinates."""

    mean, covariance = model

    transformed_mean = transform @ mean

    transformed_covariance = (
        transform
        @ covariance
        @ transform.T
    )

    return (
        transformed_mean,
        transformed_covariance,
    )


def transform_binary_model(
    model,
    transform: np.ndarray,
):
    """Transform both class distributions into new coordinates."""

    positive_model, negative_model = model

    return (
        transform_gaussian_model(
            positive_model,
            transform,
        ),
        transform_gaussian_model(
            negative_model,
            transform,
        ),
    )


# -------------------------------------------------------------
# Partition evaluation
# -------------------------------------------------------------

def collect_training_samples(
    feature_name: str,
    feature_cache: dict[
        int,
        dict[int, dict[str, np.ndarray]],
    ],
    calibration_seeds: tuple[int, ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect positive and negative calibration observations."""

    positive = []
    negative = []

    for seed in calibration_seeds:
        positive.append(
            feature_cache[seed][1][feature_name]
        )

        for depth in NORMAL_DEPTHS:
            negative.append(
                feature_cache[seed][depth][feature_name]
            )

    return positive, negative


def evaluate_model(
    model,
    feature_cache: dict[
        int,
        dict[int, dict[str, np.ndarray]],
    ],
    held_out_seeds: tuple[int, ...],
    feature_name: str,
) -> dict:
    """Evaluate one fitted model on held-out seeds."""

    tp = fn = tn = fp = 0

    seed_positive_scores = []
    seed_negative_scores = []

    log_ratios = []
    predictions = []

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

        log_ratios.append(positive_lr)
        predictions.append(positive_correct)

        negative_correct = []

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

            log_ratios.append(negative_lr)
            predictions.append(not is_correct)

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
        "seed_balanced_accuracy": seed_balanced_accuracy,
        "log_ratios": np.asarray(log_ratios),
        "predictions": np.asarray(predictions, dtype=bool),
    }


# -------------------------------------------------------------
# Single partition and regularization evaluation
# -------------------------------------------------------------

def evaluate_partition(
    feature_cache: dict[
        int,
        dict[int, dict[str, np.ndarray]],
    ],
    calibration_seeds: tuple[int, ...],
    held_out_seeds: tuple[int, ...],
    regularization: float,
) -> dict:
    """Compare independent B/C fits and affine-transformed B."""

    if set(calibration_seeds) & set(held_out_seeds):
        raise ValueError(
            "Calibration and held-out seeds must be disjoint"
        )

    # ---------------------------------------------------------
    # Fit B: training accuracy + generalization gap
    # ---------------------------------------------------------

    positive_b, negative_b = collect_training_samples(
        "training_gap",
        feature_cache,
        calibration_seeds,
    )

    model_b = fit_binary_model(
        positive_b,
        negative_b,
        regularization,
    )

    result_b = evaluate_model(
        model_b,
        feature_cache,
        held_out_seeds,
        "training_gap",
    )

    # ---------------------------------------------------------
    # Fit C independently:
    # training accuracy + evaluation accuracy
    # ---------------------------------------------------------

    positive_c, negative_c = collect_training_samples(
        "training_evaluation",
        feature_cache,
        calibration_seeds,
    )

    model_c = fit_binary_model(
        positive_c,
        negative_c,
        regularization,
    )

    result_c = evaluate_model(
        model_c,
        feature_cache,
        held_out_seeds,
        "training_evaluation",
    )

    # ---------------------------------------------------------
    # Transform the fitted B model into C coordinates.
    #
    # This transforms the already regularized covariance:
    #
    # Sigma_C = A @ Sigma_B @ A.T
    #
    # It does NOT add another isotropic regularizer.
    # ---------------------------------------------------------

    transformed_model_b = transform_binary_model(
        model_b,
        FEATURE_TRANSFORM,
    )

    transformed_result_b = evaluate_model(
        transformed_model_b,
        feature_cache,
        held_out_seeds,
        "training_evaluation",
    )

    # ---------------------------------------------------------
    # Compare B and C likelihood ratios under the affine map.
    # ---------------------------------------------------------

    lr_differences = np.abs(
        result_b["log_ratios"]
        - transformed_result_b["log_ratios"]
    )

    decision_disagreements = int(
        np.count_nonzero(
            result_b["predictions"]
            != transformed_result_b["predictions"]
        )
    )

    # Compare independently fitted C against transformed B.
    independent_decision_disagreements = int(
        np.count_nonzero(
            transformed_result_b["predictions"]
            != result_c["predictions"]
        )
    )

    return {
        "b": result_b,
        "c": result_c,
        "transformed_b": transformed_result_b,
        "max_affine_lr_error": float(
            np.max(lr_differences)
        ),
        "mean_affine_lr_error": float(
            np.mean(lr_differences)
        ),
        "affine_decision_disagreements": decision_disagreements,
        "independent_decision_disagreements": (
            independent_decision_disagreements
        ),
    }


# -------------------------------------------------------------
# Summary statistics
# -------------------------------------------------------------

def summarize(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


# -------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("H7 COVARIANCE REGULARIZATION SENSITIVITY EXPERIMENT")
    print("=" * 70)

    print(f"Seed pool size: {len(SEED_POOL)}")
    print(f"Partitions: {N_PARTITIONS}")
    print(f"Calibration seeds per partition: {CALIBRATION_SIZE}")
    print(f"Held-out seeds per partition: {HELD_OUT_SIZE}")

    print(
        "\nRegularization values:",
        REGULARIZATION_VALUES,
    )

    # ---------------------------------------------------------
    # Generate partitions
    # ---------------------------------------------------------

    partitions = generate_partitions()

    all_seeds = set()

    for calibration, held_out in partitions:
        all_seeds.update(calibration)
        all_seeds.update(held_out)

    # ---------------------------------------------------------
    # Build cache
    # ---------------------------------------------------------

    print("\nBuilding feature cache...")
    print(f"Unique seeds required: {len(all_seeds)}")

    feature_cache = build_feature_cache(
        all_seeds
    )

    print("\nFeature cache completed.")

    # ---------------------------------------------------------
    # Verify feature transformation
    # ---------------------------------------------------------

    max_feature_error = 0.0

    for seed in all_seeds:
        for depth in (1, *NORMAL_DEPTHS):
            b = feature_cache[seed][depth]["training_gap"]
            c = feature_cache[seed][depth]["training_evaluation"]

            transformed_b = FEATURE_TRANSFORM @ b

            error = float(
                np.max(np.abs(c - transformed_b))
            )

            max_feature_error = max(
                max_feature_error,
                error,
            )

    print(
        "\nMaximum feature transformation error: "
        f"{max_feature_error:.12e}"
    )

    # ---------------------------------------------------------
    # Evaluate all partitions and regularization strengths
    # ---------------------------------------------------------

    results_by_lambda: dict[float, list[dict]] = {}

    for regularization in REGULARIZATION_VALUES:
        print("\n")
        print("-" * 70)
        print(
            f"REGULARIZATION = {regularization:.0e}"
        )
        print("-" * 70)

        partition_results = []

        for partition_index, (
            calibration_seeds,
            held_out_seeds,
        ) in enumerate(partitions, start=1):

            result = evaluate_partition(
                feature_cache,
                calibration_seeds,
                held_out_seeds,
                regularization,
            )

            partition_results.append(result)

            print(
                f"Partition {partition_index:02d}: "
                f"B={result['b']['seed_balanced_accuracy']:.2%}, "
                f"C={result['c']['seed_balanced_accuracy']:.2%}, "
                f"Affine LR error="
                f"{result['max_affine_lr_error']:.2e}"
            )

        results_by_lambda[regularization] = partition_results

    # ---------------------------------------------------------
    # Summarize regularization sensitivity
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("REGULARIZATION SENSITIVITY SUMMARY")
    print("=" * 70)

    for regularization, partition_results in results_by_lambda.items():
        b_scores = [
            result["b"]["seed_balanced_accuracy"]
            for result in partition_results
        ]

        c_scores = [
            result["c"]["seed_balanced_accuracy"]
            for result in partition_results
        ]

        differences = (
            np.asarray(b_scores)
            - np.asarray(c_scores)
        )

        b_summary = summarize(b_scores)
        c_summary = summarize(c_scores)
        difference_summary = summarize(
            differences.tolist()
        )

        print(
            f"\nRegularization: {regularization:.0e}"
        )

        print(
            f"  B mean accuracy:       "
            f"{b_summary['mean']:.2%}"
        )

        print(
            f"  B standard deviation:  "
            f"{b_summary['std']:.2%}"
        )

        print(
            f"  C mean accuracy:       "
            f"{c_summary['mean']:.2%}"
        )

        print(
            f"  C standard deviation:  "
            f"{c_summary['std']:.2%}"
        )

        print(
            f"  Mean B-C difference:   "
            f"{difference_summary['mean']:+.3%}"
        )

        print(
            f"  Difference SD:         "
            f"{difference_summary['std']:.3%}"
        )

        print(
            f"  Partitions B better:   "
            f"{np.mean(differences > 0):.1%}"
        )

        max_affine_error = max(
            result["max_affine_lr_error"]
            for result in partition_results
        )

        mean_affine_error = float(
            np.mean([
                result["mean_affine_lr_error"]
                for result in partition_results
            ])
        )

        affine_disagreements = sum(
            result["affine_decision_disagreements"]
            for result in partition_results
        )

        independent_disagreements = sum(
            result["independent_decision_disagreements"]
            for result in partition_results
        )

        print(
            f"  Maximum affine LR error: "
            f"{max_affine_error:.3e}"
        )

        print(
            f"  Mean affine LR error:    "
            f"{mean_affine_error:.3e}"
        )

        print(
            f"  Affine decision disagreements: "
            f"{affine_disagreements}"
        )

        print(
            f"  Independent B/C decision disagreements: "
            f"{independent_disagreements}"
        )

    # ---------------------------------------------------------
    # Final interpretation notes
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENT NOTES")
    print("=" * 70)

    print(
        "1. B and C are mathematically related by an invertible "
        "linear transformation."
    )

    print(
        "2. Independent isotropic covariance regularization "
        "is not invariant under this transformation."
    )

    print(
        "3. The affine-transformed B model provides an "
        "invariance check."
    )

    print(
        "4. Affine likelihood-ratio errors should be close "
        "to floating-point precision."
    )

    print(
        "5. Repeated partitions reuse seeds and are not "
        "independent replications."
    )

    print(
        "6. No production likelihood or registry code is modified."
    )


if __name__ == "__main__":
    main()
