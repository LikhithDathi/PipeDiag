import pytest

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)

from experiments.generalization_calibration import (
    collect_generalization_calibration_observations,
)


def test_calibration_returns_observations():
    observations = (
        collect_generalization_calibration_observations()
    )

    assert observations


def test_expected_number_of_observations():
    observations = (
        collect_generalization_calibration_observations(
            random_states=(1, 2, 3, 4, 5),
        )
    )

    assert len(observations) == 10


def test_all_observations_are_calibration_observations():
    observations = (
        collect_generalization_calibration_observations()
    )

    assert all(
        isinstance(
            observation,
            CalibrationObservation,
        )
        for observation in observations
    )


def test_all_observations_belong_to_g2():
    observations = (
        collect_generalization_calibration_observations()
    )

    assert all(
        observation.evidence_group == "G2"
        for observation in observations
    )


def test_all_observations_target_h2():
    observations = (
        collect_generalization_calibration_observations()
    )

    assert all(
        observation.hypothesis == "H2"
        for observation in observations
    )


def test_both_true_and_false_observations_exist():
    observations = (
        collect_generalization_calibration_observations()
    )

    states = {
        observation.hypothesis_true
        for observation in observations
    }

    assert states == {True, False}


def test_true_and_false_counts_match():
    observations = (
        collect_generalization_calibration_observations(
            random_states=(1, 2, 3, 4),
        )
    )

    true_count = sum(
        observation.hypothesis_true
        for observation in observations
    )

    false_count = sum(
        not observation.hypothesis_true
        for observation in observations
    )

    assert true_count == 4
    assert false_count == 4


def test_observations_are_finite():
    observations = (
        collect_generalization_calibration_observations()
    )

    assert all(
        observation.value == pytest.approx(
            observation.value
        )
        for observation in observations
    )


def test_more_random_states_produce_more_observations():
    observations = (
        collect_generalization_calibration_observations(
            random_states=(1, 2, 3, 4, 5, 6),
        )
    )

    assert len(observations) == 12


def test_custom_normal_depth_is_supported():
    observations = (
        collect_generalization_calibration_observations(
            random_states=(1, 2, 3),
            normal_max_depth=3,
        )
    )

    assert len(observations) == 6