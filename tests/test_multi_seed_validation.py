from experiments.multi_seed_validation import (
    VALIDATION_SEEDS,
    build_multi_seed_scenarios,
    run_multi_seed_validation,
)
from pipediag.diagnostics.scenario_result import ScenarioResult


def test_validation_seeds_are_independent_from_calibration_seeds():
    assert VALIDATION_SEEDS == tuple(range(100, 110))


def test_build_multi_seed_scenarios_creates_nine_hypotheses_per_seed():
    scenarios = build_multi_seed_scenarios()

    assert len(scenarios) == 90

    for seed in VALIDATION_SEEDS:
        seed_scenarios = [
            scenario
            for scenario in scenarios
            if scenario.configuration["random_state"] == seed
        ]

        assert len(seed_scenarios) == 9


def test_build_multi_seed_scenarios_cover_all_hypotheses():
    scenarios = build_multi_seed_scenarios()

    hypotheses = {
        scenario.expected_hypothesis
        for scenario in scenarios
    }

    assert hypotheses == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
    }


def test_all_multi_seed_scenarios_are_single_fault_scenarios():
    scenarios = build_multi_seed_scenarios()

    assert all(
        scenario.metadata["scenario_type"] == "independent_single_fault"
        for scenario in scenarios
    )

    assert all(
        scenario.metadata["fault_injected"] is True
        for scenario in scenarios
    )


def test_run_multi_seed_validation_returns_scenario_results():
    validation = run_multi_seed_validation()

    assert len(validation.results) == 90
    assert all(
        isinstance(result, ScenarioResult)
        for result in validation.results
    )


def test_multi_seed_validation_counts_are_consistent():
    validation = run_multi_seed_validation()

    assert validation.total_scenarios == 90
    assert validation.correct_predictions == 86
    assert validation.incorrect_predictions == 4


def test_multi_seed_validation_accuracy():
    validation = run_multi_seed_validation()

    assert validation.top1_accuracy == 86 / 90


def test_failed_predictions_are_high_uncertainty():
    validation = run_multi_seed_validation()

    failures = [
        result
        for result in validation.results
        if result.is_correct is False
    ]

    assert len(failures) == 4

    for result in failures:
        assert result.uncertainty > 0.90


def test_results_preserve_posteriors():
    validation = run_multi_seed_validation()

    for result in validation.results:
        assert set(result.posterior) == {
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
            "H7",
            "H8",
            "H9",
        }

        assert abs(sum(result.posterior.values()) - 1.0) < 1e-9