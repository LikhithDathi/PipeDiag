import pytest

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.calibration_runner import (
    CalibrationRunner,
)
from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
)


class FakeEvidenceProvider:
    def __init__(self, evidence_by_scenario):
        self.evidence_by_scenario = evidence_by_scenario

    def collect(self, scenario):
        return list(
            self.evidence_by_scenario[
                scenario.scenario_id
            ]
        )


def create_scenario(
    scenario_id,
    expected_hypothesis,
):
    return DiagnosticScenario(
        scenario_id=scenario_id,
        expected_hypothesis=expected_hypothesis,
        description="Calibration scenario.",
    )


def test_runner_returns_calibration_observations():
    scenarios = [
        create_scenario(
            "leakage_001",
            "H1",
        )
    ]

    provider = FakeEvidenceProvider(
        {
            "leakage_001": [
                EvidenceGroup(
                    name="G1",
                    value=0.95,
                    status="observed",
                )
            ]
        }
    )

    runner = CalibrationRunner(
        evidence_provider=provider
    )

    result = runner.collect(
        scenarios=scenarios,
        hypotheses=["H1", "H2"],
    )

    assert result
    assert all(
        isinstance(
            item,
            CalibrationObservation,
        )
        for item in result
    )


def test_runner_collects_observations_from_multiple_scenarios():
    scenarios = [
        create_scenario(
            "leakage_001",
            "H1",
        ),
        create_scenario(
            "clean_001",
            None,
        ),
    ]

    provider = FakeEvidenceProvider(
        {
            "leakage_001": [
                EvidenceGroup(
                    name="G1",
                    value=0.95,
                    status="observed",
                )
            ],
            "clean_001": [
                EvidenceGroup(
                    name="G1",
                    value=0.25,
                    status="observed",
                )
            ],
        }
    )

    runner = CalibrationRunner(
        evidence_provider=provider
    )

    result = runner.collect(
        scenarios=scenarios,
        hypotheses=["H1", "H2"],
    )

    assert len(result) == 4


def test_runner_preserves_evidence_values():
    scenarios = [
        create_scenario(
            "leakage_001",
            "H1",
        )
    ]

    provider = FakeEvidenceProvider(
        {
            "leakage_001": [
                EvidenceGroup(
                    name="G1",
                    value=0.95,
                    status="observed",
                )
            ]
        }
    )

    runner = CalibrationRunner(
        evidence_provider=provider
    )

    result = runner.collect(
        scenarios=scenarios,
        hypotheses=["H1", "H2"],
    )

    assert all(
        item.value == pytest.approx(0.95)
        for item in result
    )


def test_runner_preserves_ground_truth():
    scenarios = [
        create_scenario(
            "leakage_001",
            "H1",
        )
    ]

    provider = FakeEvidenceProvider(
        {
            "leakage_001": [
                EvidenceGroup(
                    name="G1",
                    value=0.95,
                    status="observed",
                )
            ]
        }
    )

    runner = CalibrationRunner(
        evidence_provider=provider
    )

    result = runner.collect(
        scenarios=scenarios,
        hypotheses=["H1", "H2"],
    )

    h1 = next(
        item
        for item in result
        if item.hypothesis == "H1"
    )

    assert h1.hypothesis_true is True


def test_runner_marks_clean_scenario_as_negative_control():
    scenarios = [
        create_scenario(
            "clean_001",
            None,
        )
    ]

    provider = FakeEvidenceProvider(
        {
            "clean_001": [
                EvidenceGroup(
                    name="G1",
                    value=0.25,
                    status="observed",
                )
            ]
        }
    )

    runner = CalibrationRunner(
        evidence_provider=provider
    )

    result = runner.collect(
        scenarios=scenarios,
        hypotheses=["H1", "H2"],
    )

    assert len(result) == 2

    assert all(
        item.hypothesis_true is False
        for item in result
    )


def test_runner_ignores_unavailable_evidence():
    scenarios = [
        create_scenario(
            "leakage_001",
            "H1",
        )
    ]

    provider = FakeEvidenceProvider(
        {
            "leakage_001": [
                EvidenceGroup(
                    name="G1",
                    value=None,
                    status="unavailable",
                )
            ]
        }
    )

    runner = CalibrationRunner(
        evidence_provider=provider
    )

    result = runner.collect(
        scenarios=scenarios,
        hypotheses=["H1", "H2"],
    )

    assert result == []


def test_runner_rejects_empty_scenarios():
    runner = CalibrationRunner(
        evidence_provider=FakeEvidenceProvider({})
    )

    with pytest.raises(ValueError):
        runner.collect(
            scenarios=[],
            hypotheses=["H1", "H2"],
        )


def test_runner_rejects_empty_hypotheses():
    scenarios = [
        create_scenario(
            "leakage_001",
            "H1",
        )
    ]

    runner = CalibrationRunner(
        evidence_provider=FakeEvidenceProvider({})
    )

    with pytest.raises(ValueError):
        runner.collect(
            scenarios=scenarios,
            hypotheses=[],
        )


def test_runner_rejects_duplicate_scenario_ids():
    scenarios = [
        create_scenario(
            "leakage_001",
            "H1",
        ),
        create_scenario(
            "leakage_001",
            "H1",
        ),
    ]

    runner = CalibrationRunner(
        evidence_provider=FakeEvidenceProvider({})
    )

    with pytest.raises(ValueError):
        runner.collect(
            scenarios=scenarios,
            hypotheses=["H1", "H2"],
        )


def test_runner_requires_evidence_provider():
    with pytest.raises(ValueError):
        CalibrationRunner(
            evidence_provider=None
        )


def test_runner_passes_each_scenario_to_provider():
    scenarios = [
        create_scenario(
            "leakage_001",
            "H1",
        )
    ]

    provider = FakeEvidenceProvider(
        {
            "leakage_001": [
                EvidenceGroup(
                    name="G1",
                    value=0.95,
                    status="observed",
                )
            ]
        }
    )

    runner = CalibrationRunner(
        evidence_provider=provider
    )

    runner.collect(
        scenarios=scenarios,
        hypotheses=["H1", "H2"],
    )

    assert "leakage_001" in (
        provider.evidence_by_scenario
    )