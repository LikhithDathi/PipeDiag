import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.diagnostic_engine import (
    DiagnosticEngine,
)
from pipediag.diagnostics.evidence_group import (
    EvidenceGroup,
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


EVIDENCE_MAP = {
    "G1": {
        "primary": {"H1"},
        "secondary": set(),
    }
}


def _build_engine():
    registry = LikelihoodModelRegistry()

    registry.register(
        "G1",
        "H1",
        GaussianLikelihoodModel(
            mean_true=0.90,
            std_true=0.05,
            mean_false=0.10,
            std_false=0.05,
        ),
    )

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_MAP,
    )

    bayesian = BayesianInference(
        hypothesis_ids=[
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
            "H7",
            "H8",
            "H9",
        ]
    )

    evidence_inference = BayesianEvidenceInference(
        bayesian
    )

    return DiagnosticEngine(
        likelihood_evaluator=evaluator,
        bayesian_evidence_inference=evidence_inference,
    )


def test_engine_uses_likelihood_ratios():
    engine = _build_engine()

    evidence = EvidenceGroup(
        name="G1",
        value=0.90,
        status="observed",
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert result.posterior["H1"] > 1.0 / 9.0


def test_engine_reduces_support_when_evidence_favors_alternative():
    engine = _build_engine()

    evidence = EvidenceGroup(
        name="G1",
        value=0.10,
        status="observed",
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert result.posterior["H1"] < 1.0 / 9.0


def test_unavailable_evidence_preserves_prior():
    engine = _build_engine()

    evidence = EvidenceGroup(
        name="G1",
        value=0.90,
        status="unavailable",
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert result.posterior["H1"] == pytest.approx(
        1.0 / 9.0
    )


def test_engine_rejects_duplicate_evidence_groups():
    engine = _build_engine()

    evidence = EvidenceGroup(
        name="G1",
        value=0.90,
        status="observed",
    )

    with pytest.raises(ValueError):
        engine.diagnose(
            evidence_groups=[
                evidence,
                evidence,
            ]
        )


def test_engine_returns_normalized_posterior():
    engine = _build_engine()

    evidence = EvidenceGroup(
        name="G1",
        value=0.90,
        status="observed",
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

    assert sum(result.posterior.values()) == pytest.approx(
        1.0
    )


def test_engine_preserves_all_hypotheses():
    engine = _build_engine()

    evidence = EvidenceGroup(
        name="G1",
        value=0.90,
        status="observed",
    )

    result = engine.diagnose(
        evidence_groups=[evidence]
    )

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