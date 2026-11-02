import pytest

from pipediag.diagnostics.diagnostic_scenario import (
    DiagnosticScenario,
)
from pipediag.diagnostics.scenario_result import (
    ScenarioResult,
)
from pipediag.diagnostics.scenario_runner import (
    ScenarioRunner,
)
from pipediag.diagnostics.validation import (
    DiagnosticValidator,
)
from pipediag.diagnostics.validation_result import (
    ValidationResult,
)


class FakeEvidenceProvider:
    def __init__(self, evidence=None):
        self.evidence = evidence or []
        self.scenarios = []

    def collect(self, scenario):
        self.scenarios.append(scenario)
        return self.evidence


class FakeDiagnosticEngine:
    def __init__(self, posterior):
        self.posterior = posterior
        self.scenarios = []

    def diagnose(self, evidence_groups, metadata=None):
        self.scenarios.append(
            (evidence_groups, metadata)
        )

        return type(
            "FakeDiagnosticResult",
            (),
            {
                "posterior": self.posterior,
                "uncertainty": 0.25,
            },
        )()


def _scenario(
    scenario_id="scenario_1",
    expected_hypothesis="H1",
):
    return DiagnosticScenario(
        scenario_id=scenario_id,
        expected_hypothesis=expected_hypothesis,
        description="Test scenario",
    )


def _validator():
    evidence_provider = FakeEvidenceProvider(
        evidence=["evidence"]
    )

    posterior = {
        "H1": 0.8,
        "H2": 0.1,
        "H3": 0.1,
    }

    diagnostic_engine = FakeDiagnosticEngine(
        posterior
    )

    runner = ScenarioRunner(
        evidence_provider=evidence_provider,
        diagnostic_engine=diagnostic_engine,
    )

    return DiagnosticValidator(runner)


def test_validation_result_identifies_correct_prediction():
    result = ValidationResult(
        scenario_id="s1",
        expected_hypothesis="H1",
        predicted_hypothesis="H1",
        expected_posterior=0.8,
        uncertainty=0.2,
        posterior={
            "H1": 0.8,
            "H2": 0.1,
            "H3": 0.1,
        },
    )

    assert result.is_correct is True


def test_validation_result_identifies_incorrect_prediction():
    result = ValidationResult(
        scenario_id="s1",
        expected_hypothesis="H1",
        predicted_hypothesis="H2",
        expected_posterior=0.1,
        uncertainty=0.2,
        posterior={
            "H1": 0.1,
            "H2": 0.8,
            "H3": 0.1,
        },
    )

    assert result.is_correct is False


def test_clean_scenario_has_no_expected_hypothesis():
    result = ValidationResult(
        scenario_id="clean",
        expected_hypothesis=None,
        predicted_hypothesis="H1",
        expected_posterior=None,
        uncertainty=0.9,
        posterior={
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        },
    )

    assert result.is_correct is None


def test_validator_runs_scenario_runner():
    validator = _validator()

    results = validator.validate(
        [_scenario()]
    )

    assert len(results) == 1
    assert results[0].scenario_id == "scenario_1"


def test_validator_returns_expected_posterior():
    validator = _validator()

    results = validator.validate(
        [_scenario()]
    )

    assert results[0].expected_posterior == pytest.approx(
        0.8
    )


def test_validator_identifies_correct_prediction():
    validator = _validator()

    results = validator.validate(
        [_scenario()]
    )

    assert results[0].predicted_hypothesis == "H1"
    assert results[0].is_correct is True


def test_validator_preserves_full_posterior():
    validator = _validator()

    results = validator.validate(
        [_scenario()]
    )

    assert results[0].posterior == {
        "H1": 0.8,
        "H2": 0.1,
        "H3": 0.1,
    }


def test_validator_rejects_none_scenarios():
    validator = _validator()

    with pytest.raises(
        ValueError,
        match="scenarios are required",
    ):
        validator.validate(None)


def test_validator_rejects_none_inside_scenarios():
    validator = _validator()

    with pytest.raises(
        ValueError,
        match="must not contain None",
    ):
        validator.validate([None])


def test_validator_rejects_duplicate_scenario_ids():
    validator = _validator()

    scenarios = [
        _scenario("same"),
        _scenario("same"),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate scenario",
    ):
        validator.validate(scenarios)


def test_validator_supports_multiple_scenarios():
    validator = _validator()

    scenarios = [
        _scenario("s1"),
        _scenario("s2"),
        _scenario("s3"),
    ]

    results = validator.validate(scenarios)

    assert [r.scenario_id for r in results] == [
        "s1",
        "s2",
        "s3",
    ]


def test_validation_result_rejects_empty_scenario_id():
    with pytest.raises(
        ValueError,
        match="scenario_id",
    ):
        ValidationResult(
            scenario_id="",
            expected_hypothesis="H1",
            predicted_hypothesis="H1",
            expected_posterior=0.8,
            uncertainty=0.2,
            posterior={"H1": 1.0},
        )


def test_validation_result_rejects_empty_posterior():
    with pytest.raises(
        ValueError,
        match="non-empty posterior",
    ):
        ValidationResult(
            scenario_id="s1",
            expected_hypothesis="H1",
            predicted_hypothesis="H1",
            expected_posterior=1.0,
            uncertainty=0.0,
            posterior={},
        )


def test_validation_result_rejects_invalid_uncertainty():
    with pytest.raises(
        ValueError,
        match="uncertainty",
    ):
        ValidationResult(
            scenario_id="s1",
            expected_hypothesis="H1",
            predicted_hypothesis="H1",
            expected_posterior=0.8,
            uncertainty=1.5,
            posterior={"H1": 1.0},
        )


def test_validation_result_rejects_probability_outside_range():
    with pytest.raises(
        ValueError,
        match="Posterior probabilities",
    ):
        ValidationResult(
            scenario_id="s1",
            expected_hypothesis="H1",
            predicted_hypothesis="H1",
            expected_posterior=1.2,
            uncertainty=0.2,
            posterior={
                "H1": 1.2,
            },
        )


def test_validation_result_requires_expected_posterior():
    with pytest.raises(
        ValueError,
        match="expected_posterior is required",
    ):
        ValidationResult(
            scenario_id="s1",
            expected_hypothesis="H1",
            predicted_hypothesis="H1",
            expected_posterior=None,
            uncertainty=0.2,
            posterior={"H1": 1.0},
        )


def test_validation_result_rejects_unexpected_expected_posterior():
    with pytest.raises(
        ValueError,
        match="expected_posterior must be None",
    ):
        ValidationResult(
            scenario_id="s1",
            expected_hypothesis=None,
            predicted_hypothesis="H1",
            expected_posterior=1.0,
            uncertainty=0.2,
            posterior={"H1": 1.0},
        )