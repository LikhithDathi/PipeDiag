from pathlib import Path

from pipediag.diagnostics.calibrated_registry import (
    load_calibrated_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "likelihood_models.json"
)


def test_load_calibrated_registry():
    registry = load_calibrated_registry(ARTIFACT_PATH)

    models = registry.models()

    assert len(models) == 11

    assert ("G1", "H1", "default") in models
    assert ("G2", "H2", "generalization_gap") in models
    assert ("G2", "H7", "training_accuracy") in models
    assert ("G3", "H3", "mean_absolute_standardized_shift") in models
    assert ("G4", "H4", "minority_class_proportion") in models
    assert ("G5", "H5", "missing_value_rate") in models
    assert ("G5", "H7", "missing_value_rate") in models
    assert ("G6", "H6", "log_wilson_interval_half_width") in models
    assert ("G7", "H8", "constant_feature_ratio") in models
    assert ("G8", "H9", "execution_failure_rate") in models


import pytest


def test_loaded_model_parameters_match_artifact():
    registry = load_calibrated_registry(ARTIFACT_PATH)

    model = registry.get("G2", "H2")

    assert model is not None
    assert model.measurement == "generalization_gap"

    assert model.mean_true == pytest.approx(
        0.17466666666666666
    )
    assert model.std_true == pytest.approx(
        0.046671428328499034
    )
    assert model.mean_false == pytest.approx(
        0.04125
    )
    assert model.std_false == pytest.approx(
        0.03639339454717938
    )




def test_load_calibrated_registry_uses_default_artifact_path():
    registry = load_calibrated_registry()

    assert len(registry.models()) == 11