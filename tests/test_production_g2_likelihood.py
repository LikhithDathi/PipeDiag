import pytest

from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.likelihood_builder import (
    LikelihoodModelBuilder,
)

from experiments.generalization_calibration import (
    collect_generalization_calibration_observations,
)
from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.underfitting_calibration import (
    collect_underfitting_calibration_observations,
)
from experiments.overfitting_pipeline import (
    run_overfitting_pipeline,
)


def _build_evaluator():
    h2_observations = (
        collect_generalization_calibration_observations(
            n_samples=500,
            n_features=10,
            random_states=(1, 2, 3, 4, 5),
        )
    )

    h7_observations = (
        collect_underfitting_calibration_observations(
            n_samples=500,
            n_features=10,
            random_states=(1, 2, 3, 4, 5),
        )
    )

    registry = LikelihoodModelBuilder().build(
        h2_observations + h7_observations
    )

    return EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )


def test_production_g2_map_contains_h2_and_h7():
    mapping = EVIDENCE_HYPOTHESIS_MAP["G2"]

    hypotheses = (
        mapping["primary"]
        | mapping["secondary"]
    )

    assert "H2" in hypotheses
    assert "H7" in hypotheses


def test_production_g2_evaluator_returns_h2_and_h7():
    evaluator = _build_evaluator()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=1,
    )

    evidence = collect_generalization_evidence(
        result
    )

    ratios = evaluator.evaluate_ratios(evidence)

    assert set(ratios) == {"H2", "H7"}


def test_production_g2_ratios_are_positive_and_finite():
    evaluator = _build_evaluator()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=1,
    )

    evidence = collect_generalization_evidence(
        result
    )

    ratios = evaluator.evaluate_ratios(evidence)

    for ratio in ratios.values():
        assert ratio > 0
        assert ratio == pytest.approx(
            float(ratio)
        )


def test_h7_ratio_uses_training_accuracy():
    evaluator = _build_evaluator()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=1,
    )

    evidence = collect_generalization_evidence(
        result
    )

    ratios = evaluator.evaluate_ratios(evidence)

    h7_ratio = ratios["H7"]

    h7_model = evaluator.registry.get(
        "G2",
        "H7",
    )

    training_accuracy = evidence.metadata[
        "measurements"
    ]["training_accuracy"].value

    expected_ratio = h7_model.likelihood_ratio(
        training_accuracy
    )

    assert h7_ratio == pytest.approx(
        expected_ratio
    )


def test_h2_ratio_uses_generalization_gap():
    evaluator = _build_evaluator()

    result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        max_depth=None,
    )

    evidence = collect_generalization_evidence(
        result
    )

    ratios = evaluator.evaluate_ratios(evidence)

    h2_ratio = ratios["H2"]

    h2_model = evaluator.registry.get(
        "G2",
        "H2",
    )

    expected_ratio = h2_model.likelihood_ratio(
        evidence.value
    )

    assert h2_ratio == pytest.approx(
        expected_ratio
    )