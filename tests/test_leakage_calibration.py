import pytest

from experiments.leakage_calibration import (
    collect_leakage_calibration_observations,
)


def test_collects_clean_and_leakage_observations():
    observations = collect_leakage_calibration_observations(
        random_states=(1, 2, 3),
    )

    assert len(observations) == 6

    true_observations = [
        observation
        for observation in observations
        if observation.hypothesis_true
    ]

    false_observations = [
        observation
        for observation in observations
        if not observation.hypothesis_true
    ]

    assert len(true_observations) == 3
    assert len(false_observations) == 3


def test_all_observations_are_for_g1_h1():
    observations = collect_leakage_calibration_observations(
        random_states=(1, 2),
    )

    assert all(
        observation.evidence_group == "G1"
        for observation in observations
    )

    assert all(
        observation.hypothesis == "H1"
        for observation in observations
    )


def test_observations_contain_finite_values():
    observations = collect_leakage_calibration_observations(
        random_states=(1, 2, 3),
    )

    assert all(
        observation.value >= 0.0
        for observation in observations
    )


def test_clean_observations_are_negative_controls():
    observations = collect_leakage_calibration_observations(
        random_states=(1, 2, 3),
    )

    clean_observations = [
        observation
        for observation in observations
        if not observation.hypothesis_true
    ]

    assert len(clean_observations) == 3


def test_leakage_observations_are_positive_controls():
    observations = collect_leakage_calibration_observations(
        random_states=(1, 2, 3),
    )

    leakage_observations = [
        observation
        for observation in observations
        if observation.hypothesis_true
    ]

    assert len(leakage_observations) == 3


def test_leakage_evidence_is_stronger_than_clean_evidence():
    observations = collect_leakage_calibration_observations(
        random_states=(1, 2, 3, 4, 5),
    )

    clean_values = [
        observation.value
        for observation in observations
        if not observation.hypothesis_true
    ]

    leakage_values = [
        observation.value
        for observation in observations
        if observation.hypothesis_true
    ]

    assert sum(leakage_values) / len(leakage_values) > (
        sum(clean_values) / len(clean_values)
    )


def test_number_of_observations_matches_random_states():
    random_states = (10, 20, 30, 40)

    observations = collect_leakage_calibration_observations(
        random_states=random_states,
    )

    assert len(observations) == 2 * len(random_states)


def test_rejects_invalid_sample_count():
    with pytest.raises(ValueError):
        collect_leakage_calibration_observations(
            n_samples=0,
        )


def test_rejects_invalid_feature_count():
    with pytest.raises(ValueError):
        collect_leakage_calibration_observations(
            n_features=0,
        )


def test_rejects_empty_random_states():
    with pytest.raises(ValueError):
        collect_leakage_calibration_observations(
            random_states=(),
        )