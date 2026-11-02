import pytest

from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)


def test_scenario_stores_required_information():
    scenario = DiagnosticScenario(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        description="Target information is present in the features.",
    )

    assert scenario.scenario_id == "leakage_001"
    assert scenario.expected_hypothesis == "H1"
    assert (
        scenario.description
        == "Target information is present in the features."
    )


def test_scenario_defaults_configuration_to_empty_dictionary():
    scenario = DiagnosticScenario(
        scenario_id="clean_001",
        expected_hypothesis=None,
        description="Clean reference pipeline.",
    )

    assert scenario.configuration == {}


def test_scenario_defaults_metadata_to_empty_dictionary():
    scenario = DiagnosticScenario(
        scenario_id="clean_001",
        expected_hypothesis=None,
        description="Clean reference pipeline.",
    )

    assert scenario.metadata == {}


def test_scenario_rejects_empty_scenario_id():
    with pytest.raises(ValueError):
        DiagnosticScenario(
            scenario_id="",
            expected_hypothesis="H1",
            description="Test scenario.",
        )


def test_scenario_rejects_empty_description():
    with pytest.raises(ValueError):
        DiagnosticScenario(
            scenario_id="scenario_001",
            expected_hypothesis="H1",
            description="",
        )


def test_scenario_allows_no_expected_hypothesis():
    scenario = DiagnosticScenario(
        scenario_id="clean_001",
        expected_hypothesis=None,
        description="Clean reference pipeline.",
    )

    assert scenario.expected_hypothesis is None


def test_scenario_rejects_empty_expected_hypothesis():
    with pytest.raises(ValueError):
        DiagnosticScenario(
            scenario_id="scenario_001",
            expected_hypothesis="",
            description="Test scenario.",
        )


def test_scenario_preserves_configuration():
    configuration = {
        "model": "logistic_regression",
        "train_size": 800,
        "test_size": 200,
    }

    scenario = DiagnosticScenario(
        scenario_id="scenario_001",
        expected_hypothesis="H2",
        description="Generalization failure scenario.",
        configuration=configuration,
    )

    assert scenario.configuration == configuration


def test_scenario_preserves_metadata():
    metadata = {
        "dataset": "synthetic_v1",
        "seed": 42,
    }

    scenario = DiagnosticScenario(
        scenario_id="scenario_001",
        expected_hypothesis="H2",
        description="Generalization failure scenario.",
        metadata=metadata,
    )

    assert scenario.metadata == metadata


def test_scenario_does_not_require_mutable_input():
    configuration = {
        "seed": 42,
    }

    scenario = DiagnosticScenario(
        scenario_id="scenario_001",
        expected_hypothesis="H2",
        description="Test scenario.",
        configuration=configuration,
    )

    configuration["seed"] = 100

    assert scenario.configuration["seed"] == 42