import math

import pytest

from pipediag.acquisition.evaluation import EvaluationEvidence
from pipediag.diagnostics.adapters import (
    diagnostic_evidence_to_group,
    generalization_to_evidence,
)
from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import BayesianEvidenceInference
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.generalization import (
    HIGHER_IS_BETTER,
    GeneralizationDiagnostic,
)
from pipediag.diagnostics.likelihood_model import GaussianLikelihoodModel
from pipediag.diagnostics.likelihood_registry import LikelihoodModelRegistry


HYPOTHESES = [f"H{i}" for i in range(1, 10)]


def make_g2():
    evaluations = [
        EvaluationEvidence(
            metric_name="accuracy",
            metric_value=0.96,
            evaluation_context="TRAIN",
            feature_dataset_id="D-TRAIN-X",
            target_dataset_id="D-TRAIN-Y",
            sample_count=100,
            source="explicit_metric",
        ),
        EvaluationEvidence(
            metric_name="accuracy",
            metric_value=0.61,
            evaluation_context="TEST",
            feature_dataset_id="D-TEST-X",
            target_dataset_id="D-TEST-Y",
            sample_count=100,
            source="explicit_metric",
        ),
    ]

    diagnostic = GeneralizationDiagnostic()

    generalization = diagnostic.analyze(
        evaluations,
        metric_direction=HIGHER_IS_BETTER,
    )

    evidence = generalization_to_evidence(generalization)

    return diagnostic_evidence_to_group(evidence)


def make_registry():
    registry = LikelihoodModelRegistry()

    # These are deliberately simple test calibration parameters.
    #
    # They are NOT production calibration values.
    #
    # H2: larger generalization gaps are more likely.
    registry.register(
        "G2",
        "H2",
        GaussianLikelihoodModel(
            mean_true=0.30,
            std_true=0.10,
            mean_false=0.02,
            std_false=0.05,
            measurement="gap",
        ),
    )

    # H7: smaller gaps are more typical for underfitting.
    registry.register(
        "G2",
        "H7",
        GaussianLikelihoodModel(
            mean_true=0.05,
            std_true=0.05,
            mean_false=0.20,
            std_false=0.10,
            measurement="gap",
        ),
    )

    return registry


def test_real_g2_evidence_reaches_likelihood_evaluator():
    group = make_g2()
    registry = make_registry()

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    ratios = evaluator.evaluate_ratios(group)

    assert set(ratios) == {"H2", "H7"}

    assert all(
        isinstance(value, float) and math.isfinite(value)
        for value in ratios.values()
    )

    # The observed gap of 0.35 should provide stronger support
    # for H2 than H7 under these test calibration parameters.
    assert ratios["H2"] > ratios["H7"]


def test_real_g2_evidence_produces_valid_bayesian_posterior():
    group = make_g2()
    registry = make_registry()

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    bayesian = BayesianEvidenceInference(
        BayesianInference(HYPOTHESES)
    )

    log_ratios = evaluator.evaluate_log_ratios(group)

    posterior = bayesian.infer_with_likelihood_ratios(
        {
            group.name: {
                hypothesis: math.exp(log_ratio)
                for hypothesis, log_ratio in log_ratios.items()
            }
        }
    )

    assert set(posterior) == set(HYPOTHESES)

    assert all(
        isinstance(value, float)
        and math.isfinite(value)
        and value >= 0
        for value in posterior.values()
    )

    assert sum(posterior.values()) == pytest.approx(1.0)

    assert posterior["H2"] > posterior["H7"]


def test_real_g2_evidence_has_expected_measurement():
    group = make_g2()

    assert group.name == "G2"
    assert group.status == "observed"

    measurements = group.metadata["measurements"]

    assert measurements["train_value"] == pytest.approx(0.96)
    assert measurements["test_value"] == pytest.approx(0.61)
    assert measurements["gap"] == pytest.approx(0.35)