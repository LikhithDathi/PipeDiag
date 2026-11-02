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

    hypotheses = [
        "H1",
        "H2",
        "H3",
    ]

    bayesian = BayesianInference(
        hypotheses
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

    bayesian_evidence = (
        BayesianEvidenceInference(
            bayesian
        )
    )

    return DiagnosticEngine(
        likelihood_evaluator=evaluator,
        bayesian_evidence_inference=(
            bayesian_evidence
        ),
    )


def test_engine_returns_diagnostic_result():
    engine = create_engine()

    result = engine.diagnose(
        [
            EvidenceGroup(
                name="G1",
                value=0.8,
                status="observed",
            )
        ]
    )

    assert result.posterior
    assert 0.0 <= result.uncertainty <= 1.0


def test_engine_preserves_evidence():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        )
    ]

    result = engine.diagnose(evidence)

    assert result.evidence == evidence


def test_engine_returns_prior_when_evidence_unavailable():
    engine = create_engine()

    result = engine.diagnose(
        [
            EvidenceGroup(
                name="G1",
                value=None,
                status="unavailable",
            )
        ]
    )

    assert result.posterior == pytest.approx(
        {
            "H1": 1 / 3,
            "H2": 1 / 3,
            "H3": 1 / 3,
        }
    )


def test_engine_preserves_metadata():
    engine = create_engine()

    metadata = {
        "run_id": "run-001"
    }

    result = engine.diagnose(
        [
            EvidenceGroup(
                name="G1",
                value=0.8,
                status="observed",
            )
        ],
        metadata=metadata,
    )

    assert result.metadata == metadata


def test_engine_defaults_metadata_to_empty_dictionary():
    engine = create_engine()

    result = engine.diagnose(
        [
            EvidenceGroup(
                name="G1",
                value=0.8,
                status="observed",
            )
        ]
    )

    assert result.metadata == {}


def test_engine_does_not_modify_evidence():
    engine = create_engine()

    evidence = [
        EvidenceGroup(
            name="G1",
            value=0.8,
            status="observed",
        )
    ]

    original = list(evidence)

    engine.diagnose(evidence)

    assert evidence == original


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