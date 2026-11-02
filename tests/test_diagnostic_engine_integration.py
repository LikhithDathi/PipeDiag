import pytest

from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import (
    BayesianEvidenceInference,
)
from pipediag.diagnostics.diagnostic_engine import DiagnosticEngine
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


HYPOTHESES = ["H1", "H2", "H3"]


def create_engine():
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

    registry.register(
        "G2",
        "H2",
        GaussianLikelihoodModel(
            mean_true=0.8,
            std_true=0.1,
            mean_false=0.2,
            std_false=0.1,
        ),
    )

    registry.register(
        "G2",
        "H3",
        GaussianLikelihoodModel(
            mean_true=0.6,
            std_true=0.1,
            mean_false=0.2,
            std_false=0.1,
        ),
    )

    bayesian = BayesianInference(HYPOTHESES)

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map={
            "G1": {
                "primary": {"H1"},
                "secondary": set(),
            },
            "G2": {
                "primary": {"H2", "H3"},
                "secondary": set(),
            },
        },
    )

    bayesian_evidence = BayesianEvidenceInference(
        bayesian
    )

    return DiagnosticEngine(
        likelihood_evaluator=evaluator,
        bayesian_evidence_inference=bayesian_evidence,
    )


def test_engine_produces_diagnostic_result():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        )
    ]

    result = engine.diagnose(evidence)

    assert result.posterior
    assert result.uncertainty >= 0.0
    assert result.uncertainty <= 1.0
    assert result.evidence == evidence


def test_engine_uses_evidence_to_update_posterior():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        )
    ]

    result = engine.diagnose(evidence)

    assert result.posterior["H1"] > result.posterior["H2"]
    assert result.posterior["H1"] > result.posterior["H3"]


def test_engine_combines_multiple_evidence_groups():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        ),
        EvidenceGroup(
            name="G2",
            value=0.8,
            status="observed",
        ),
    ]

    result = engine.diagnose(evidence)

    assert result.posterior["H1"] > 0.0
    assert result.posterior["H2"] > 0.0
    assert result.posterior["H3"] > 0.0


def test_engine_ignores_unavailable_evidence():
    engine = create_engine()

    observed = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        )
    ]

    with_unavailable = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        ),
        EvidenceGroup(
            name="G2",
            value=0.8,
            status="unavailable",
        ),
    ]

    result_observed = engine.diagnose(observed)
    result_with_unavailable = engine.diagnose(
        with_unavailable
    )

    assert result_with_unavailable.posterior == pytest.approx(
        result_observed.posterior
    )


def test_engine_returns_prior_when_no_evidence_is_available():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=None,
            status="unavailable",
        )
    ]

    result = engine.diagnose(evidence)

    assert result.posterior == pytest.approx(
        {
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        }
    )


def test_engine_posterior_sums_to_one():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        )
    ]

    result = engine.diagnose(evidence)

    assert sum(result.posterior.values()) == pytest.approx(
        1.0
    )


def test_engine_preserves_metadata():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        )
    ]

    metadata = {
        "run_id": "run-001",
        "model": "test-model",
    }

    result = engine.diagnose(
        evidence,
        metadata=metadata,
    )

    assert result.metadata == metadata


def test_engine_rejects_duplicate_evidence_groups():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        ),
        EvidenceGroup(
            name="G1",
            value=0.7,
            status="observed",
        ),
    ]

    with pytest.raises(ValueError):
        engine.diagnose(evidence)