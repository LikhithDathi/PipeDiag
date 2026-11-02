import pytest

from pipediag.diagnostics.scenario_result import (
    ScenarioResult,
)


def test_result_stores_required_information():
    result = ScenarioResult(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        posterior={
            "H1": 0.8,
            "H2": 0.1,
            "H3": 0.1,
        },
        uncertainty=0.3,
    )

    assert result.scenario_id == "leakage_001"
    assert result.expected_hypothesis == "H1"
    assert result.uncertainty == 0.3


def test_predicted_hypothesis_is_highest_posterior():
    result = ScenarioResult(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        posterior={
            "H1": 0.8,
            "H2": 0.1,
            "H3": 0.1,
        },
        uncertainty=0.3,
    )

    assert result.predicted_hypothesis == "H1"


def test_result_identifies_correct_prediction():
    result = ScenarioResult(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        posterior={
            "H1": 0.8,
            "H2": 0.1,
            "H3": 0.1,
        },
        uncertainty=0.3,
    )

    assert result.is_correct is True


def test_result_identifies_incorrect_prediction():
    result = ScenarioResult(
        scenario_id="underfitting_001",
        expected_hypothesis="H7",
        posterior={
            "H1": 0.7,
            "H7": 0.2,
            "H2": 0.1,
        },
        uncertainty=0.4,
    )

    assert result.predicted_hypothesis == "H1"
    assert result.is_correct is False


def test_clean_scenario_has_no_expected_hypothesis():
    result = ScenarioResult(
        scenario_id="clean_001",
        expected_hypothesis=None,
        posterior={
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        },
        uncertainty=1.0,
    )

    assert result.expected_hypothesis is None
    assert result.is_correct is None


def test_posterior_must_not_be_empty():
    with pytest.raises(ValueError):
        ScenarioResult(
            scenario_id="scenario_001",
            expected_hypothesis="H1",
            posterior={},
            uncertainty=0.5,
        )


def test_posterior_must_sum_to_one():
    with pytest.raises(ValueError):
        ScenarioResult(
            scenario_id="scenario_001",
            expected_hypothesis="H1",
            posterior={
                "H1": 0.6,
                "H2": 0.2,
            },
            uncertainty=0.5,
        )


def test_posterior_probabilities_must_be_valid():
    with pytest.raises(ValueError):
        ScenarioResult(
            scenario_id="scenario_001",
            expected_hypothesis="H1",
            posterior={
                "H1": 1.2,
                "H2": -0.2,
            },
            uncertainty=0.5,
        )


def test_uncertainty_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        ScenarioResult(
            scenario_id="scenario_001",
            expected_hypothesis="H1",
            posterior={
                "H1": 0.7,
                "H2": 0.3,
            },
            uncertainty=1.5,
        )


def test_result_preserves_metadata():
    result = ScenarioResult(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        posterior={
            "H1": 0.8,
            "H2": 0.2,
        },
        uncertainty=0.3,
        metadata={
            "seed": 42,
            "dataset": "synthetic_v1",
        },
    )

    assert result.metadata == {
        "seed": 42,
        "dataset": "synthetic_v1",
    }