import math

import pytest

from experiments.clean_pipeline import run_clean_pipeline
from experiments.leakage_calibration import (
    collect_leakage_calibration_observations,
)
from experiments.leakage_evidence import collect_leakage_evidence
from experiments.leakage_pipeline import run_leakage_pipeline

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.calibration_estimator import (
    CalibrationEstimator,
)
from pipediag.diagnostics.hypotheses import (
    DIAGNOSTIC_HYPOTHESES,
)
from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)


def _build_calibrated_inference():
    observations = collect_leakage_calibration_observations(
        n_samples=500,
        n_features=10,
        random_states=(1, 2, 3, 4, 5, 6),
        leakage_strengths=(
            0.80,
            0.84,
            0.88,
            0.92,
            0.96,
            0.90,
        ),
    )

    parameters = CalibrationEstimator().fit(
        observations
    )

    model = GaussianLikelihoodModel(
        mean_true=parameters.mean_true,
        std_true=parameters.std_true,
        mean_false=parameters.mean_false,
        std_false=parameters.std_false,
    )

    bayesian = BayesianInference(
        hypothesis_ids=[
            hypothesis.id
            for hypothesis in DIAGNOSTIC_HYPOTHESES
        ]
    )

    inference = BayesianEvidenceInference(
        bayesian_inference=bayesian,
    )

    return model, inference


def test_calibrated_model_produces_finite_likelihoods():
    model, _ = _build_calibrated_inference()

    pipeline_result = run_leakage_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        leakage_strength=0.90,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    likelihood = model.likelihood(
        evidence_value=float(evidence.value),
        hypothesis_true=True,
    )

    assert math.isfinite(likelihood)
    assert likelihood >= 0.0


def test_leakage_evidence_increases_h1_posterior():
    model, inference = _build_calibrated_inference()

    pipeline_result = run_leakage_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
        leakage_strength=0.90,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    likelihood_true = model.likelihood(
        evidence_value=float(evidence.value),
        hypothesis_true=True,
    )

    posterior = inference.infer(
        {
            evidence.name: {
                "H1": likelihood_true,
            }
        }
    )

    assert posterior["H1"] > 1.0 / len(
        DIAGNOSTIC_HYPOTHESES
    )


def test_clean_evidence_has_lower_likelihood_under_h1():
    model, _ = _build_calibrated_inference()

    pipeline_result = run_clean_pipeline(
        n_samples=500,
        n_features=10,
        random_state=100,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    likelihood_true = model.likelihood(
        evidence_value=float(evidence.value),
        hypothesis_true=True,
    )

    likelihood_false = model.likelihood(
        evidence_value=float(evidence.value),
        hypothesis_true=False,
    )

    assert likelihood_true < likelihood_false


def test_posterior_is_normalized():
    model, inference = _build_calibrated_inference()

    pipeline_result = run_leakage_pipeline(
        n_samples=500,
        n_features=10,
        random_state=101,
        leakage_strength=0.90,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    likelihood_true = model.likelihood(
        evidence_value=float(evidence.value),
        hypothesis_true=True,
    )

    posterior = inference.infer(
        {
            evidence.name: {
                "H1": likelihood_true,
            }
        }
    )

    assert sum(posterior.values()) == pytest.approx(
        1.0
    )


def test_all_hypotheses_remain_in_posterior():
    model, inference = _build_calibrated_inference()

    pipeline_result = run_leakage_pipeline(
        n_samples=500,
        n_features=10,
        random_state=102,
        leakage_strength=0.90,
    )

    evidence = collect_leakage_evidence(
        pipeline_result
    )

    likelihood_true = model.likelihood(
        evidence_value=float(evidence.value),
        hypothesis_true=True,
    )

    posterior = inference.infer(
        {
            evidence.name: {
                "H1": likelihood_true,
            }
        }
    )

    assert set(posterior) == {
        hypothesis.id
        for hypothesis in DIAGNOSTIC_HYPOTHESES
    }
