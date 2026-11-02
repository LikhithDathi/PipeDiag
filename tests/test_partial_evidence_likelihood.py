import pytest

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.likelihood_model import GaussianLikelihoodModel
from pipediag.diagnostics.likelihood_registry import (
    LikelihoodModelRegistry,
)


CALIBRATED_EVIDENCE_MAP = {
    "G1": {
        "primary": {"H1"},
        "secondary": set(),
    }
}


def _build_evaluator():
    registry = LikelihoodModelRegistry()

    registry.register(
        "G1",
        "H1",
        GaussianLikelihoodModel(
            mean_true=0.95,
            std_true=0.05,
            mean_false=0.10,
            std_false=0.05,
        ),
    )

    return EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=CALIBRATED_EVIDENCE_MAP,
    )


def test_evaluator_returns_available_calibrated_relationships():
    evaluator = _build_evaluator()

    evidence = EvidenceGroup(
        name="G1",
        value=0.95,
        status="observed",
    )

    likelihoods = evaluator.evaluate(evidence)

    assert "H1" in likelihoods
    assert len(likelihoods) == 1


def test_observed_evidence_requires_registered_model():
    registry = LikelihoodModelRegistry()

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=CALIBRATED_EVIDENCE_MAP,
    )

    evidence = EvidenceGroup(
        name="G1",
        value=0.95,
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)


def test_unavailable_evidence_returns_empty_likelihoods():
    evaluator = _build_evaluator()

    evidence = EvidenceGroup(
        name="G1",
        value=0.95,
        status="unavailable",
    )

    assert evaluator.evaluate(evidence) == {}


def test_unknown_evidence_group_is_rejected():
    evaluator = _build_evaluator()

    evidence = EvidenceGroup(
        name="G99",
        value=0.95,
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)


def test_invalid_evidence_status_is_rejected():
    evaluator = _build_evaluator()

    evidence = EvidenceGroup(
        name="G1",
        value=0.95,
        status="invalid",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)


def test_non_numeric_evidence_is_rejected():
    evaluator = _build_evaluator()

    evidence = EvidenceGroup(
        name="G1",
        value="high",
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)


def test_non_finite_evidence_is_rejected():
    evaluator = _build_evaluator()

    evidence = EvidenceGroup(
        name="G1",
        value=float("nan"),
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(evidence)




def test_evaluator_can_return_likelihood_ratios():
    evaluator = _build_evaluator()

    evidence = EvidenceGroup(
        name="G1",
        value=0.95,
        status="observed",
    )

    ratios = evaluator.evaluate_ratios(evidence)

    assert "H1" in ratios
    assert ratios["H1"] > 1.0


def test_unavailable_evidence_returns_empty_ratios():
    evaluator = _build_evaluator()

    evidence = EvidenceGroup(
        name="G1",
        value=0.95,
        status="unavailable",
    )

    assert evaluator.evaluate_ratios(evidence) == {}


def test_ratio_evaluation_rejects_missing_model():
    registry = LikelihoodModelRegistry()

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=CALIBRATED_EVIDENCE_MAP,
    )

    evidence = EvidenceGroup(
        name="G1",
        value=0.95,
        status="observed",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate_ratios(evidence)