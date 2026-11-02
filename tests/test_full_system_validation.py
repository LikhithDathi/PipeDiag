import pytest

from experiments.full_system_validation import (
    HYPOTHESIS_IDS,
    FullSystemValidation,
    SingleFaultEvidenceProvider,
    build_calibrated_likelihood_registry,
    build_diagnostic_engine,
    build_single_fault_scenarios,
    run_single_fault_validation,
)


def test_hypothesis_ids_contain_all_nine_hypotheses():
    assert HYPOTHESIS_IDS == (
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
    )


def test_build_single_fault_scenarios_contains_all_hypotheses():
    scenarios = build_single_fault_scenarios()

    assert len(scenarios) == 9

    assert [
        scenario.expected_hypothesis
        for scenario in scenarios
    ] == list(HYPOTHESIS_IDS)


def test_single_fault_scenario_ids_are_unique():
    scenarios = build_single_fault_scenarios()

    ids = [
        scenario.scenario_id
        for scenario in scenarios
    ]

    assert len(ids) == len(set(ids))


def test_single_fault_scenarios_have_expected_hypotheses():
    scenarios = build_single_fault_scenarios()

    for scenario in scenarios:
        assert scenario.expected_hypothesis in HYPOTHESIS_IDS


def test_calibrated_registry_contains_models():
    registry = build_calibrated_likelihood_registry()

    assert registry.models()


def test_diagnostic_engine_can_be_built():
    engine = build_diagnostic_engine()

    assert engine is not None


def test_single_fault_provider_collects_h1():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[0]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G1"
    assert evidence[0].status == "observed"


def test_single_fault_provider_collects_h2():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[1]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G2"


def test_single_fault_provider_collects_h3():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[2]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G3"


def test_single_fault_provider_collects_h4():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[3]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G4"


def test_single_fault_provider_collects_h5():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[4]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G5"


def test_single_fault_provider_collects_h6():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[5]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G6"


def test_single_fault_provider_collects_h7():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[6]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G2"


def test_single_fault_provider_collects_h8():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[7]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G7"


def test_single_fault_provider_collects_h9():
    provider = SingleFaultEvidenceProvider()

    scenario = build_single_fault_scenarios()[8]

    evidence = provider.collect(scenario)

    assert len(evidence) == 1
    assert evidence[0].name == "G8"


def test_single_fault_validation_returns_nine_results():
    validation = run_single_fault_validation()

    assert isinstance(
        validation,
        FullSystemValidation,
    )

    assert validation.total_scenarios == 9


def test_single_fault_posteriors_are_complete():
    validation = run_single_fault_validation()

    for result in validation.results:
        assert set(result.posterior) == set(
            HYPOTHESIS_IDS
        )


def test_single_fault_posteriors_are_normalized():
    validation = run_single_fault_validation()

    for result in validation.results:
        assert sum(
            result.posterior.values()
        ) == pytest.approx(1.0)


def test_single_fault_predictions_are_known_hypotheses():
    validation = run_single_fault_validation()

    for result in validation.results:
        assert result.predicted_hypothesis in (
            HYPOTHESIS_IDS
        )


def test_single_fault_uncertainty_is_valid():
    validation = run_single_fault_validation()

    for result in validation.results:
        assert 0.0 <= result.uncertainty <= 1.0


def test_full_system_validation_top1_accuracy_matches_results():
    validation = run_single_fault_validation()

    expected_correct = sum(
        result.is_correct is True
        for result in validation.results
    )

    assert validation.correct_predictions == (
        expected_correct
    )

    assert validation.top1_accuracy == (
        expected_correct / 9
    )


def test_single_fault_validation_has_expected_scenario_order():
    validation = run_single_fault_validation()

    assert [
        result.expected_hypothesis
        for result in validation.results
    ] == list(HYPOTHESIS_IDS)