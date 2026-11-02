import pytest

from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.evidence_measurement import (
    EvidenceMeasurement,
)
from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)
from pipediag.diagnostics.likelihood_registry import (
    LikelihoodModelRegistry,
)


CALIBRATED_MAP = {
    "G2": {
        "primary": {"H2", "H7"},
        "secondary": set(),
    }
}


def _build_registry():
    registry = LikelihoodModelRegistry()

    registry.register(
        "G2",
        "H2",
        GaussianLikelihoodModel(
            mean_true=0.30,
            std_true=0.05,
            mean_false=0.05,
            std_false=0.03,
            measurement="generalization_gap",
        ),
    )

    registry.register(
        "G2",
        "H7",
        GaussianLikelihoodModel(
            mean_true=0.70,
            std_true=0.05,
            mean_false=0.90,
            std_false=0.05,
            measurement="training_accuracy",
        ),
    )

    return registry


def _build_evidence():
    return EvidenceGroup(
        name="G2",
        value=0.30,
        status="observed",
        metadata={
            "measurements": {
                "generalization_gap": EvidenceMeasurement(
                    name="generalization_gap",
                    value=0.30,
                ),
                "training_accuracy": EvidenceMeasurement(
                    name="training_accuracy",
                    value=0.70,
                ),
            }
        },
    )


def test_evaluator_uses_h2_measurement():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=_build_registry(),
        evidence_hypothesis_map=CALIBRATED_MAP,
    )

    evidence = _build_evidence()

    ratios = evaluator.evaluate_ratios(evidence)

    assert ratios["H2"] > 1.0


def test_evaluator_uses_h7_measurement():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=_build_registry(),
        evidence_hypothesis_map=CALIBRATED_MAP,
    )

    evidence = _build_evidence()

    ratios = evaluator.evaluate_ratios(evidence)

    assert ratios["H7"] > 1.0


def test_h7_does_not_use_generalization_gap():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=_build_registry(),
        evidence_hypothesis_map=CALIBRATED_MAP,
    )

    evidence = _build_evidence()

    ratios = evaluator.evaluate_ratios(evidence)

    h7_model = _build_registry().get(
        "G2",
        "H7",
    )

    wrong_ratio = h7_model.likelihood_ratio(
        evidence.value
    )

    assert ratios["H7"] != pytest.approx(
        wrong_ratio
    )


def test_missing_required_measurement_is_rejected():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=_build_registry(),
        evidence_hypothesis_map=CALIBRATED_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.30,
        status="observed",
        metadata={
            "measurements": {
                "generalization_gap": EvidenceMeasurement(
                    name="generalization_gap",
                    value=0.30,
                )
            }
        },
    )

    with pytest.raises(ValueError):
        evaluator.evaluate_ratios(evidence)


def test_default_measurement_uses_evidence_value():
    registry = LikelihoodModelRegistry()

    registry.register(
        "G1",
        "H1",
        GaussianLikelihoodModel(
            mean_true=0.8,
            std_true=0.1,
            mean_false=0.2,
            std_false=0.1,
        ),
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map={
            "G1": {
                "primary": {"H1"},
                "secondary": set(),
            }
        },
    )

    evidence = EvidenceGroup(
        name="G1",
        value=0.8,
        status="observed",
    )

    ratios = evaluator.evaluate_ratios(
        evidence
    )

    assert ratios["H1"] > 1.0


def test_unavailable_evidence_returns_empty_result():
    evaluator = EvidenceLikelihoodEvaluator(
        registry=_build_registry(),
        evidence_hypothesis_map=CALIBRATED_MAP,
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.30,
        status="unavailable",
    )

    assert evaluator.evaluate_ratios(evidence) == {}