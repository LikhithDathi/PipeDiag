import pytest

from pipediag.diagnostics.diagnostic_engine import DiagnosticEngine
from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.scenario_runner import (
    ScenarioRunner,
)
from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
)


class FakeEvidenceProvider:
    def __init__(self, evidence):
        self.evidence = evidence
        self.received_scenario = None

    def collect(self, scenario):
        self.received_scenario = scenario
        return list(self.evidence)


class FakeDiagnosticEngine:
    def __init__(self):
        self.received_evidence = None
        self.received_metadata = None

    def diagnose(self, evidence_groups, metadata=None):
        self.received_evidence = list(evidence_groups)
        self.received_metadata = metadata

        from pipediag.diagnostics.diagnostic_result import (
            DiagnosticResult,
        )

        return DiagnosticResult(
            posterior={
                "H1": 0.8,
                "H2": 0.1,
                "H3": 0.1,
            },
            uncertainty=0.4,
            evidence=list(evidence_groups),
            metadata=metadata or {},
        )


def create_scenario():
    return DiagnosticScenario(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        description="Target information is present in the features.",
    )


def create_evidence():
    return [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        )
    ]


def test_runner_returns_scenario_result():
    scenario = create_scenario()

    provider = FakeEvidenceProvider(
        create_evidence()
    )

    engine = FakeDiagnosticEngine()

    runner = ScenarioRunner(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    result = runner.run(scenario)

    assert result.scenario_id == "leakage_001"
    assert result.expected_hypothesis == "H1"
    assert result.posterior == {
        "H1": 0.8,
        "H2": 0.1,
        "H3": 0.1,
    }


def test_runner_passes_scenario_to_evidence_provider():
    scenario = create_scenario()

    provider = FakeEvidenceProvider(
        create_evidence()
    )

    engine = FakeDiagnosticEngine()

    runner = ScenarioRunner(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    runner.run(scenario)

    assert provider.received_scenario is scenario


def test_runner_passes_evidence_to_diagnostic_engine():
    scenario = create_scenario()
    evidence = create_evidence()

    provider = FakeEvidenceProvider(evidence)
    engine = FakeDiagnosticEngine()

    runner = ScenarioRunner(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    runner.run(scenario)

    assert engine.received_evidence == evidence


def test_runner_preserves_scenario_metadata():
    scenario = DiagnosticScenario(
        scenario_id="leakage_001",
        expected_hypothesis="H1",
        description="Leakage scenario.",
        metadata={
            "seed": 42,
        },
    )

    provider = FakeEvidenceProvider(
        create_evidence()
    )

    engine = FakeDiagnosticEngine()

    runner = ScenarioRunner(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    result = runner.run(scenario)

    assert result.metadata["seed"] == 42


def test_runner_marks_correct_prediction():
    scenario = create_scenario()

    provider = FakeEvidenceProvider(
        create_evidence()
    )

    engine = FakeDiagnosticEngine()

    runner = ScenarioRunner(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    result = runner.run(scenario)

    assert result.predicted_hypothesis == "H1"
    assert result.is_correct is True


def test_runner_supports_clean_scenario():
    scenario = DiagnosticScenario(
        scenario_id="clean_001",
        expected_hypothesis=None,
        description="Clean reference pipeline.",
    )

    provider = FakeEvidenceProvider(
        create_evidence()
    )

    engine = FakeDiagnosticEngine()

    runner = ScenarioRunner(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    result = runner.run(scenario)

    assert result.expected_hypothesis is None
    assert result.is_correct is None


def test_runner_rejects_missing_evidence_provider():
    with pytest.raises(ValueError):
        ScenarioRunner(
            evidence_provider=None,
            diagnostic_engine=FakeDiagnosticEngine(),
        )


def test_runner_rejects_missing_diagnostic_engine():
    with pytest.raises(ValueError):
        ScenarioRunner(
            evidence_provider=FakeEvidenceProvider(
                create_evidence()
            ),
            diagnostic_engine=None,
        )


def test_runner_rejects_missing_scenario():
    runner = ScenarioRunner(
        evidence_provider=FakeEvidenceProvider(
            create_evidence()
        ),
        diagnostic_engine=FakeDiagnosticEngine(),
    )

    with pytest.raises(ValueError):
        runner.run(None)


def test_runner_preserves_diagnostic_metadata():
    scenario = create_scenario()

    provider = FakeEvidenceProvider(
        create_evidence()
    )

    engine = FakeDiagnosticEngine()

    runner = ScenarioRunner(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    result = runner.run(scenario)

    assert result.metadata == {}