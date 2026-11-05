import pytest

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)
from pipediag.diagnostics.likelihood_registry import (
    LikelihoodModelRegistry,
)


def create_model(mean_true=0.8, mean_false=0.2):
    return GaussianLikelihoodModel(
        mean_true=mean_true,
        std_true=0.1,
        mean_false=mean_false,
        std_false=0.1,
    )


def create_registry():
    registry = LikelihoodModelRegistry()

    registry.register(
        "G2",
        "H2",
        create_model(mean_true=0.8),
    )

    registry.register(
        "G2",
        "H7",
        create_model(mean_true=0.7),
    )

    return registry


def test_evaluator_returns_likelihood_for_relevant_hypotheses():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.8,
        status="observed",
    )

    result = evaluator.evaluate(evidence)

    assert "H2" in result
    assert "H7" in result


def test_evaluator_does_not_require_models_for_unrelated_hypotheses():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.8,
        status="observed",
    )

    result = evaluator.evaluate(evidence)

    assert "H1" not in result
    assert "H3" not in result
    assert "H4" not in result
    assert "H5" not in result
    assert "H6" not in result
    assert "H8" not in result
    assert "H9" not in result


def test_higher_true_density_is_reflected_in_result():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.8,
        status="observed",
    )

    result = evaluator.evaluate(evidence)

    assert result["H2"] > result["H7"]


def test_unavailable_evidence_returns_empty_result():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.8,
        status="unavailable",
    )

    assert evaluator.evaluate(evidence) == {}


def test_unknown_evidence_group_is_rejected():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G99",
        value=0.8,
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)


def test_observed_evidence_requires_registered_model():
    registry = LikelihoodModelRegistry()

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.8,
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)


def test_evidence_value_must_be_numeric():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value="invalid",
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)


def test_non_finite_evidence_value_is_rejected():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=float("nan"),
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)


def test_result_contains_density_values():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.8,
        status="observed",
    )

    result = evaluator.evaluate(evidence)

    assert result["H2"] > 0.0
    assert result["H7"] > 0.0

@pytest.mark.parametrize(
    "method_name",
    [
        "evaluate",
        "evaluate_ratios",
        "evaluate_log_ratios",
    ],
)
def test_inference_ineligible_evidence_is_skipped(method_name):
    evaluator = EvidenceLikelihoodEvaluator(
        registry=create_registry(),
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value="CONFIRMED_LEAKAGE",
        status="observed",
        metadata={
            "inference_eligible": False,
        },
    )

    result = getattr(evaluator, method_name)(evidence)

    assert result == {}


def test_numeric_g1_evidence_remains_eligible_for_inference():
    registry = LikelihoodModelRegistry()
    registry.register(
        "G1",
        "H1",
        create_model(),
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    evidence = EvidenceGroup(
        name="G1",
        value=0.8,
        status="observed",
    )

    result = evaluator.evaluate(evidence)

    assert "H1" in result
    assert result["H1"] > 0.0
