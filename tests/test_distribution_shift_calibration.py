import pytest

from experiments.distribution_shift_calibration import (
    collect_distribution_shift_calibration_observations,
)


def test_calibration_returns_two_observations_per_random_state():
    observations = (
        collect_distribution_shift_calibration_observations(
            random_states=(1, 2, 3),
        )
    )

    assert len(observations) == 6


def test_calibration_contains_only_h3_relationship():
    observations = (
        collect_distribution_shift_calibration_observations(
            random_states=(1, 2, 3),
        )
    )

    assert {
        (
            observation.evidence_group,
            observation.hypothesis,
        )
        for observation in observations
    } == {("G3", "H3")}


def test_calibration_contains_clean_and_fault_observations():
    observations = (
        collect_distribution_shift_calibration_observations(
            random_states=(1, 2, 3),
        )
    )

    assert sum(
        not observation.hypothesis_true
        for observation in observations
    ) == 3

    assert sum(
        observation.hypothesis_true
        for observation in observations
    ) == 3


def test_default_calibration_has_twenty_runs_per_class():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    assert len(observations) == 40

    assert sum(
        not observation.hypothesis_true
        for observation in observations
    ) == 20

    assert sum(
        observation.hypothesis_true
        for observation in observations
    ) == 20


def test_clean_calibration_values_are_non_negative():
    observations = (
        collect_distribution_shift_calibration_observations(
            random_states=(1, 2, 3, 4, 5),
        )
    )

    clean_values = [
        observation.value
        for observation in observations
        if not observation.hypothesis_true
    ]

    assert all(value >= 0.0 for value in clean_values)


def test_shifted_calibration_values_are_positive():
    observations = (
        collect_distribution_shift_calibration_observations(
            random_states=(1, 2, 3, 4, 5),
        )
    )

    shifted_values = [
        observation.value
        for observation in observations
        if observation.hypothesis_true
    ]

    assert all(value > 0.0 for value in shifted_values)


def test_shifted_values_are_higher_than_clean_values_on_average():
    observations = (
        collect_distribution_shift_calibration_observations()
    )

    clean_values = [
        observation.value
        for observation in observations
        if not observation.hypothesis_true
    ]

    shifted_values = [
        observation.value
        for observation in observations
        if observation.hypothesis_true
    ]

    clean_mean = sum(clean_values) / len(clean_values)
    shifted_mean = sum(shifted_values) / len(shifted_values)

    assert shifted_mean > clean_mean


def test_measurement_is_correct():
    observations = (
        collect_distribution_shift_calibration_observations(
            random_states=(1, 2),
        )
    )

    assert {
        observation.measurement
        for observation in observations
    } == {
        "mean_absolute_standardized_shift"
    }


def test_empty_random_states_are_rejected():
    with pytest.raises(ValueError, match="random_states"):
        collect_distribution_shift_calibration_observations(
            random_states=(),
        )


def test_mismatched_normal_magnitudes_are_rejected():
    with pytest.raises(
        ValueError,
        match="normal_shift_magnitudes",
    ):
        collect_distribution_shift_calibration_observations(
            random_states=(1, 2, 3),
            normal_shift_magnitudes=(0.0, 0.0),
        )


def test_mismatched_shifted_magnitudes_are_rejected():
    with pytest.raises(
        ValueError,
        match="shifted_magnitudes",
    ):
        collect_distribution_shift_calibration_observations(
            random_states=(1, 2, 3),
            shifted_magnitudes=(0.5, 0.5),
        )


def test_negative_normal_shift_is_rejected():
    with pytest.raises(
        ValueError,
        match="normal_shift_magnitudes",
    ):
        collect_distribution_shift_calibration_observations(
            random_states=(1,),
            normal_shift_magnitudes=(-0.1,),
        )


def test_non_positive_shifted_magnitude_is_rejected():
    with pytest.raises(
        ValueError,
        match="shifted_magnitudes",
    ):
        collect_distribution_shift_calibration_observations(
            random_states=(1,),
            shifted_magnitudes=(0.0,),
        )