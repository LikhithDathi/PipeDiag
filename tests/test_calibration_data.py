import pytest

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.calibration_data import (
    build_calibration_observations,
)
from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
)


HYPOTHESES = [
    "H1",
    "H2",
    "H3",
]


def test_builds_observation_for_relevant_hypothesis():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        )
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    h1 = [
        item
        for item in result
        if item.hypothesis == "H1"
    ]

    assert len(h1) == 1
    assert h1[0].hypothesis_true is True


def test_marks_other_hypotheses_false():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        )
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    false_items = [
        item
        for item in result
        if item.hypothesis != "H1"
    ]

    assert all(
        item.hypothesis_true is False
        for item in false_items
    )


def test_preserves_evidence_value():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        )
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    assert all(
        item.value == pytest.approx(0.95)
        for item in result
    )


def test_preserves_evidence_group():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        )
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    assert all(
        item.evidence_group == "G1"
        for item in result
    )


def test_unavailable_evidence_is_ignored():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=None,
            status="unavailable",
        )
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    assert result == []


def test_multiple_observed_groups_produce_multiple_observations():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        ),
        EvidenceGroup(
            name="G2",
            value=0.80,
            status="observed",
        ),
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    assert len(result) == 6


def test_each_group_has_one_observation_per_hypothesis():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        ),
        EvidenceGroup(
            name="G2",
            value=0.80,
            status="observed",
        ),
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    for group in ["G1", "G2"]:
        group_items = [
            item
            for item in result
            if item.evidence_group == group
        ]

        assert len(group_items) == 3


def test_clean_scenario_marks_all_hypotheses_false():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.25,
            status="observed",
        )
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis=None,
        hypotheses=HYPOTHESES,
    )

    assert len(result) == 3

    assert all(
        item.hypothesis_true is False
        for item in result
    )


def test_returns_calibration_observations():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        )
    ]

    result = build_calibration_observations(
        evidence_groups=evidence,
        expected_hypothesis="H1",
        hypotheses=HYPOTHESES,
    )

    assert all(
        isinstance(
            item,
            CalibrationObservation,
        )
        for item in result
    )


def test_rejects_empty_evidence_groups():
    with pytest.raises(ValueError):
        build_calibration_observations(
            evidence_groups=[],
            expected_hypothesis="H1",
            hypotheses=HYPOTHESES,
        )


def test_rejects_duplicate_evidence_groups():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        ),
        EvidenceGroup(
            name="G1",
            value=0.90,
            status="observed",
        ),
    ]

    with pytest.raises(ValueError):
        build_calibration_observations(
            evidence_groups=evidence,
            expected_hypothesis="H1",
            hypotheses=HYPOTHESES,
        )


def test_rejects_unknown_expected_hypothesis():
    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.95,
            status="observed",
        )
    ]

    with pytest.raises(ValueError):
        build_calibration_observations(
            evidence_groups=evidence,
            expected_hypothesis="H9",
            hypotheses=HYPOTHESES,
        )