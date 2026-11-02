from pathlib import Path

from pipediag.diagnostics.engine_factory import (
    build_calibrated_diagnostic_engine,
    build_diagnostic_engine,
)
from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)
from pipediag.diagnostics.likelihood_registry import (
    LikelihoodModelRegistry,
)


def test_build_diagnostic_engine():
    registry = LikelihoodModelRegistry()

    registry.register(
        "G2",
        "H2",
        GaussianLikelihoodModel(
            mean_true=0.30,
            std_true=0.10,
            mean_false=0.02,
            std_false=0.05,
            measurement="generalization_gap",
        ),
    )

    registry.register(
        "G2",
        "H7",
        GaussianLikelihoodModel(
            mean_true=0.05,
            std_true=0.05,
            mean_false=0.20,
            std_false=0.10,
            measurement="generalization_gap",
        ),
    )

    engine = build_diagnostic_engine(registry)

    assert engine is not None
    assert engine.likelihood_evaluator.registry is registry

    hypothesis_ids = (
        engine.bayesian_evidence_inference
        .bayesian_inference
        .hypothesis_ids
    )

    assert set(hypothesis_ids) == {
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


def test_build_calibrated_diagnostic_engine():
    project_root = Path(__file__).resolve().parents[1]

    artifact_path = (
        project_root
        / "artifacts"
        / "calibration"
        / "likelihood_models.json"
    )

    engine = build_calibrated_diagnostic_engine(artifact_path)

    assert engine is not None

    registry = engine.likelihood_evaluator.registry

    assert registry is not None
    assert len(registry.models()) == 11