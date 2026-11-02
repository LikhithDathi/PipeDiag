import math

import pytest

from pipediag.diagnostics.likelihood_estimator import (
    GaussianLikelihoodEstimator,
)


def test_estimator_computes_true_mean():
    estimator = GaussianLikelihoodEstimator()

    result = estimator.fit(
        true_values=[0.8, 0.9, 1.0],
        false_values=[0.1, 0.2, 0.3],
    )

    assert result.mean_true == pytest.approx(0.9)


def test_estimator_computes_false_mean():
    estimator = GaussianLikelihoodEstimator()

    result = estimator.fit(
        true_values=[0.8, 0.9, 1.0],
        false_values=[0.1, 0.2, 0.3],
    )

    assert result.mean_false == pytest.approx(0.2)


def test_estimator_computes_true_standard_deviation():
    estimator = GaussianLikelihoodEstimator()

    result = estimator.fit(
        true_values=[0.8, 0.9, 1.0],
        false_values=[0.1, 0.2, 0.3],
    )

    expected = math.sqrt(
        (
            (0.8 - 0.9) ** 2
            + (0.9 - 0.9) ** 2
            + (1.0 - 0.9) ** 2
        ) / 3
    )

    assert result.std_true == pytest.approx(expected)


def test_estimator_computes_false_standard_deviation():
    estimator = GaussianLikelihoodEstimator()

    result = estimator.fit(
        true_values=[0.8, 0.9, 1.0],
        false_values=[0.1, 0.2, 0.3],
    )

    expected = math.sqrt(
        (
            (0.1 - 0.2) ** 2
            + (0.2 - 0.2) ** 2
            + (0.3 - 0.2) ** 2
        ) / 3
    )

    assert result.std_false == pytest.approx(expected)


def test_estimator_returns_parameters_usable_by_likelihood_model():
    estimator = GaussianLikelihoodEstimator()

    result = estimator.fit(
        true_values=[0.8, 0.9, 1.0],
        false_values=[0.1, 0.2, 0.3],
    )

    assert result.mean_true == pytest.approx(0.9)
    assert result.mean_false == pytest.approx(0.2)
    assert result.std_true > 0.0
    assert result.std_false > 0.0


def test_true_values_must_not_be_empty():
    estimator = GaussianLikelihoodEstimator()

    with pytest.raises(ValueError):
        estimator.fit(
            true_values=[],
            false_values=[0.1, 0.2, 0.3],
        )


def test_false_values_must_not_be_empty():
    estimator = GaussianLikelihoodEstimator()

    with pytest.raises(ValueError):
        estimator.fit(
            true_values=[0.8, 0.9, 1.0],
            false_values=[],
        )


def test_non_finite_true_values_are_rejected():
    estimator = GaussianLikelihoodEstimator()

    with pytest.raises(ValueError):
        estimator.fit(
            true_values=[0.8, float("nan"), 1.0],
            false_values=[0.1, 0.2, 0.3],
        )


def test_non_finite_false_values_are_rejected():
    estimator = GaussianLikelihoodEstimator()

    with pytest.raises(ValueError):
        estimator.fit(
            true_values=[0.8, 0.9, 1.0],
            false_values=[0.1, float("inf"), 0.3],
        )


def test_zero_variance_true_values_are_rejected():
    estimator = GaussianLikelihoodEstimator()

    with pytest.raises(ValueError):
        estimator.fit(
            true_values=[0.8, 0.8, 0.8],
            false_values=[0.1, 0.2, 0.3],
        )


def test_zero_variance_false_values_are_rejected():
    estimator = GaussianLikelihoodEstimator()

    with pytest.raises(ValueError):
        estimator.fit(
            true_values=[0.8, 0.9, 1.0],
            false_values=[0.2, 0.2, 0.2],
        )


def test_single_sample_is_rejected():
    estimator = GaussianLikelihoodEstimator()

    with pytest.raises(ValueError):
        estimator.fit(
            true_values=[0.8],
            false_values=[0.1, 0.2],
        )


def test_estimator_does_not_modify_input_data():
    estimator = GaussianLikelihoodEstimator()

    true_values = [0.8, 0.9, 1.0]
    false_values = [0.1, 0.2, 0.3]

    original_true = true_values.copy()
    original_false = false_values.copy()

    estimator.fit(
        true_values=true_values,
        false_values=false_values,
    )

    assert true_values == original_true
    assert false_values == original_false