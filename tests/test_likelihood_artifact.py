from pathlib import Path

import pytest

from pipediag.diagnostics.likelihood_artifact import (
    load_likelihood_registry,
)


def write_artifact(
    path: Path,
    *,
    models: list[dict],
    model_count: int | None = None,
    schema_version: int = 1,
):
    import json

    payload = {
        "schema_version": schema_version,
        "model_count": (
            len(models)
            if model_count is None
            else model_count
        ),
        "models": models,
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def make_model(
    evidence_group="G2",
    hypothesis="H2",
    measurement="generalization_gap",
):
    return {
        "evidence_group": evidence_group,
        "hypothesis": hypothesis,
        "measurement": measurement,
        "mean_true": 0.30,
        "std_true": 0.10,
        "mean_false": 0.02,
        "std_false": 0.05,
    }


def test_load_likelihood_registry(tmp_path):
    artifact = tmp_path / "likelihood_models.json"

    write_artifact(
        artifact,
        models=[
            make_model("G2", "H2", "generalization_gap"),
            make_model("G2", "H7", "training_accuracy"),
        ],
    )

    registry = load_likelihood_registry(artifact)

    models = registry.models()

    assert len(models) == 2

    h2 = models[("G2", "H2")]

    assert h2.measurement == "generalization_gap"
    assert h2.mean_true == pytest.approx(0.30)
    assert h2.std_true == pytest.approx(0.10)
    assert h2.mean_false == pytest.approx(0.02)
    assert h2.std_false == pytest.approx(0.05)

    h7 = models[("G2", "H7")]

    assert h7.measurement == "training_accuracy"


def test_load_rejects_missing_artifact(tmp_path):
    artifact = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="likelihood artifact not found",
    ):
        load_likelihood_registry(artifact)


def test_load_rejects_invalid_json(tmp_path):
    artifact = tmp_path / "invalid.json"

    artifact.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid likelihood artifact JSON",
    ):
        load_likelihood_registry(artifact)


def test_load_rejects_wrong_schema_version(tmp_path):
    artifact = tmp_path / "likelihood_models.json"

    write_artifact(
        artifact,
        models=[],
        schema_version=999,
    )

    with pytest.raises(
        ValueError,
        match="unsupported likelihood artifact schema version",
    ):
        load_likelihood_registry(artifact)


def test_load_rejects_model_count_mismatch(tmp_path):
    artifact = tmp_path / "likelihood_models.json"

    write_artifact(
        artifact,
        models=[make_model()],
        model_count=99,
    )

    with pytest.raises(
        ValueError,
        match="model_count does not match",
    ):
        load_likelihood_registry(artifact)


def test_load_rejects_missing_model_field(tmp_path):
    artifact = tmp_path / "likelihood_models.json"

    model = make_model()
    del model["std_true"]

    write_artifact(
        artifact,
        models=[model],
    )

    with pytest.raises(
        ValueError,
        match="is missing fields",
    ):
        load_likelihood_registry(artifact)