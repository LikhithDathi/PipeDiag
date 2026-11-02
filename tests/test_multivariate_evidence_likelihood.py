
import math

import pytest

from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.evidence_likelihood import EvidenceLikelihoodEvaluator
from pipediag.diagnostics.likelihood_registry import LikelihoodModelRegistry
from pipediag.diagnostics.multivariate_likelihood_model import (
    MultivariateGaussianLikelihoodModel,
)


@pytest.fixture
def model():
    return MultivariateGaussianLikelihoodModel(
        mean_true=(0.70, 0.02),
        covariance_true=(
            (0.01, 0.001),
            (0.001, 0.002),
        ),
        mean_false=(0.86, 0.06),
        covariance_false=(
            (0.01, 0.001),
            (0.001, 0.002),
        ),
        measurements=("training_accuracy", "generalization_gap"),
    )


@pytest.fixture
def setup_evaluator(model):
    registry = LikelihoodModelRegistry()
    registry.register("G2", "H7", model)

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map={
            "G2": {
                "primary": {"H7"},
                "secondary": set(),
            }
        },
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.75,
        status="observed",
        metadata={
            "measurements": {
                "training_accuracy": 0.75,
                "generalization_gap": 0.025,
            }
        },
    )

    return evaluator, evidence


def test_evaluate_supports_multivariate_model(setup_evaluator):
    evaluator, evidence = setup_evaluator

    result = evaluator.evaluate(evidence)

    assert "H7" in result
    assert math.isfinite(result["H7"])
    assert result["H7"] > 0


def test_evaluate_ratios_supports_multivariate_model(setup_evaluator):
    evaluator, evidence = setup_evaluator

    result = evaluator.evaluate_ratios(evidence)

    assert "H7" in result
    assert math.isfinite(result["H7"])
    assert result["H7"] > 0


def test_evaluate_log_ratios_supports_multivariate_model(
    setup_evaluator,
):
    evaluator, evidence = setup_evaluator

    result = evaluator.evaluate_log_ratios(evidence)

    assert "H7" in result
    assert math.isfinite(result["H7"])


def test_log_ratio_matches_log_of_ratio(setup_evaluator):
    evaluator, evidence = setup_evaluator

    ratio = evaluator.evaluate_ratios(evidence)["H7"]
    log_ratio = evaluator.evaluate_log_ratios(evidence)["H7"]

    assert log_ratio == pytest.approx(math.log(ratio), abs=1e-8)


def test_missing_multivariate_measurement_raises(model):
    registry = LikelihoodModelRegistry()
    registry.register("G2", "H7", model)

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map={
            "G2": {
                "primary": {"H7"},
                "secondary": set(),
            }
        },
    )

    evidence = EvidenceGroup(
        name="G2",
        value=0.75,
        status="observed",
        metadata={
            "measurements": {
                "training_accuracy": 0.75,
            }
        },
    )

    with pytest.raises(ValueError, match="generalization_gap"):
        evaluator.evaluate_log_ratios(evidence)
