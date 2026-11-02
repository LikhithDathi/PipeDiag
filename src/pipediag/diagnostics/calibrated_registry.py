
from __future__ import annotations

import json
import sys
from pathlib import Path

from .likelihood_model import GaussianLikelihoodModel
from .likelihood_registry import LikelihoodModelRegistry


CALIBRATION_SCHEMA_VERSION = 1


def default_calibration_path() -> Path:
    """Return the calibration artifact from the source tree or installed package."""

    # 1. Artifact included inside the installed Python package.
    package_path = (
        Path(__file__).resolve().parents[1]
        / "calibration"
        / "likelihood_models.json"
    )

    if package_path.is_file():
        return package_path

    # 2. Artifact in the source repository.
    source_path = (
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "calibration"
        / "likelihood_models.json"
    )

    if source_path.is_file():
        return source_path

    # 3. Artifact installed as a shared data file.
    installed_path = (
        Path(sys.prefix)
        / "share"
        / "pipediag"
        / "calibration"
        / "likelihood_models.json"
    )

    if installed_path.is_file():
        return installed_path

    raise FileNotFoundError(
        "Calibration artifact was not found. "
        "Pass its path explicitly or reinstall PipeDiag with the artifact included."
    )


def load_calibrated_registry(
    path: str | Path | None = None,
) -> LikelihoodModelRegistry:
    """Load calibrated likelihood models from a JSON artifact."""

    artifact_path = (
        Path(path) if path is not None else default_calibration_path()
    )

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Calibration artifact was not found: {artifact_path}"
        )

    try:
        artifact = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid calibration artifact JSON: {artifact_path}"
        ) from exc

    if not isinstance(artifact, dict):
        raise ValueError("Calibration artifact must contain a JSON object")

    schema_version = artifact.get("schema_version")
    if schema_version != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported calibration artifact schema version: "
            f"{schema_version}"
        )

    models = artifact.get("models")
    if not isinstance(models, list):
        raise ValueError("Calibration artifact 'models' must be a list")

    model_count = artifact.get("model_count")
    if model_count != len(models):
        raise ValueError(
            "Calibration artifact model_count does not match models length"
        )

    registry = LikelihoodModelRegistry()

    for entry in models:
        if not isinstance(entry, dict):
            raise ValueError(
                "Each calibration model entry must be a JSON object"
            )

        required_fields = {
            "evidence_group",
            "hypothesis",
            "measurement",
            "mean_true",
            "std_true",
            "mean_false",
            "std_false",
        }

        missing_fields = required_fields - entry.keys()
        if missing_fields:
            raise ValueError(
                "Calibration model is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

        model = GaussianLikelihoodModel(
            mean_true=entry["mean_true"],
            std_true=entry["std_true"],
            mean_false=entry["mean_false"],
            std_false=entry["std_false"],
            measurement=entry["measurement"],
        )

        registry.register(
            evidence_group=entry["evidence_group"],
            hypothesis=entry["hypothesis"],
            model=model,
        )

    if len(registry.models()) != len(models):
        raise ValueError(
            "Calibration artifact contains duplicate model keys"
        )

    return registry
