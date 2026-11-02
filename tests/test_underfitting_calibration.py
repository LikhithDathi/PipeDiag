import pytest

from experiments.underfitting_calibration import (
    collect_underfitting_calibration_observations,
)


def _build_observations():
    return collect_underfitting_calibration_observations(
        n_samples=500,
        n_features=10,
        random_states=(1, 2, 3, 4, 5),
    )


def test_returns_observations():
    observations = _build_observations()

    assert observations


def test_returns_expected_number_of_observations():
    observations = _build_observations()

    # One underfitting observation and four normal
    # observations for each random state.
    assert len(observations) == 25


def test_all_observations_belong_to_g2():
    observations = _build_observations()

    assert all(
        observation.evidence_group == "G2"
        for observation in observations
    )


def test_all_observations_target_h7():
    observations = _build_observations()

    assert all(
        observation.hypothesis == "H7"
        for observation in observations
    )


def test_contains_true_and_false_observations():
    observations = _build_observations()

    values = {
        observation.hypothesis_true
        for observation in observations
    }

    assert values == {True, False}


def test_expected_true_false_counts():
    observations = _build_observations()

    true_count = sum(
        observation.hypothesis_true
        for observation in observations
    )

    false_count = sum(
        not observation.hypothesis_true
        for observation in observations
    )

    assert true_count == 5
    assert false_count == 20


def test_all_values_are_finite():
    observations = _build_observations()

    for observation in observations:
        assert observation.value == pytest.approx(
            float(observation.value)
        )


def test_underfitting_has_lower_mean_training_accuracy():
    observations = _build_observations()

    true_values = [
        observation.value
        for observation in observations
        if observation.hypothesis_true
    ]

    false_values = [
        observation.value
        for observation in observations
        if not observation.hypothesis_true
    ]

    true_mean = sum(true_values) / len(true_values)
    false_mean = sum(false_values) / len(false_values)

    assert true_mean < false_mean


def test_underfitting_depth_must_not_be_normal_depth():
    with pytest.raises(ValueError):
        collect_underfitting_calibration_observations(
            normal_max_depths=(1, 2, 3),
            underfitting_max_depth=1,
        )


def test_normal_depths_must_be_positive():
    with pytest.raises(ValueError):
        collect_underfitting_calibration_observations(
            normal_max_depths=(2, 3, 0, 5),
        )


def test_default_configuration_is_reproducible():
    first = _build_observations()
    second = _build_observations()

    assert first == second