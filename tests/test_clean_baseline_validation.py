from experiments.clean_baseline_validation import (
    CleanBaselineValidation,
    build_clean_baseline_engine,
    build_clean_baseline_scenario,
    run_clean_baseline_validation,
)


def test_build_clean_baseline_scenario_is_negative_control():
    scenario = build_clean_baseline_scenario()

    assert scenario.scenario_id == "clean_baseline"
    assert scenario.expected_hypothesis is None
    assert scenario.metadata["scenario_type"] == "negative_control"
    assert scenario.metadata["fault_injected"] is False


def test_build_clean_baseline_scenario_preserves_random_state():
    scenario = build_clean_baseline_scenario(
        random_state=123,
    )

    assert scenario.configuration["random_state"] == 123


def test_clean_baseline_engine_can_be_built():
    engine = build_clean_baseline_engine()

    assert engine is not None


def test_run_clean_baseline_returns_validation():
    validation = run_clean_baseline_validation()

    assert isinstance(
        validation,
        CleanBaselineValidation,
    )


def test_clean_baseline_contains_all_hypotheses():
    validation = run_clean_baseline_validation()

    assert set(validation.posterior) == {
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


def test_clean_baseline_posterior_is_normalized():
    validation = run_clean_baseline_validation()

    assert abs(sum(validation.posterior.values()) - 1.0) < 1e-9


def test_clean_baseline_has_no_dominant_hypothesis():
    validation = run_clean_baseline_validation()

    assert validation.maximum_posterior < 0.5
    assert validation.has_dominant_hypothesis is False


def test_clean_baseline_retains_substantial_uncertainty():
    validation = run_clean_baseline_validation()

    assert validation.uncertainty > 0.5
    assert validation.is_uncertain is True


def test_clean_baseline_passes_negative_control():
    validation = run_clean_baseline_validation()

    assert validation.passed is True


def test_clean_baseline_prediction_is_not_treated_as_ground_truth():
    validation = run_clean_baseline_validation()

    assert validation.predicted_hypothesis in {
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


def test_clean_baseline_result_preserves_negative_control_metadata():
    validation = run_clean_baseline_validation()

    assert validation.result.metadata["scenario_type"] == "negative_control"
    assert validation.result.metadata["fault_injected"] is False