import pytest

from pipediag.diagnostics.ground_truth import (
    GroundTruth,
    create_ground_truth,
)


HYPOTHESES = [
    "H1",
    "H2",
    "H3",
]


def test_ground_truth_stores_information():
    truth = GroundTruth(
        scenario_id="leakage_001",
        hypothesis="H1",
        is_true=True,
    )

    assert truth.scenario_id == "leakage_001"
    assert truth.hypothesis == "H1"
    assert truth.is_true is True


def test_creates_ground_truth_for_expected_hypothesis():
    result = create_ground_truth(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    assert len(result) == 3


def test_expected_hypothesis_is_true():
    result = create_ground_truth(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    h1 = next(
        item
        for item in result
        if item.hypothesis == "H1"
    )

    assert h1.is_true is True


def test_other_hypotheses_are_false():
    result = create_ground_truth(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    false_hypotheses = [
        item
        for item in result
        if item.hypothesis != "H1"
    ]

    assert all(
        item.is_true is False
        for item in false_hypotheses
    )


def test_ground_truth_preserves_scenario_id():
    result = create_ground_truth(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    assert all(
        item.scenario_id == "leakage_001"
        for item in result
    )


def test_clean_scenario_has_no_failure_ground_truth():
    result = create_ground_truth(
        scenario_id="clean_001",
        expected_hypothesis=None,
        hypotheses=HYPOTHESES,
    )

    assert result == []


def test_rejects_empty_scenario_id():
    with pytest.raises(ValueError):
        create_ground_truth(
            scenario_id="",
            expected_hypothesis="H1",
            hypotheses=HYPOTHESES,
        )


def test_rejects_empty_hypothesis_list():
    with pytest.raises(ValueError):
        create_ground_truth(
            scenario_id="scenario_001",
            expected_hypothesis="H1",
            hypotheses=[],
        )


def test_rejects_duplicate_hypotheses():
    with pytest.raises(ValueError):
        create_ground_truth(
            scenario_id="scenario_001",
            expected_hypothesis="H1",
            hypotheses=[
                "H1",
                "H1",
            ],
        )


def test_rejects_unknown_expected_hypothesis():
    with pytest.raises(ValueError):
        create_ground_truth(
            scenario_id="scenario_001",
            expected_hypothesis="H9",
            hypotheses=HYPOTHESES,
        )


def test_ground_truth_is_immutable():
    truth = GroundTruth(
        scenario_id="scenario_001",
        hypothesis="H1",
        is_true=True,
    )

    with pytest.raises(AttributeError):
        truth.is_true = False