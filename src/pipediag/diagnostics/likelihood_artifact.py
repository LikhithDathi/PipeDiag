from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .likelihood_model import GaussianLikelihoodModel
from .likelihood_registry import LikelihoodModelRegistry


SCHEMA_VERSION = 1


def load_likelihood_registry(
    path: str | Path,
) -> LikelihoodModelRegistry:
    """
    Load calibrated likelihood models from a persisted JSON artifact.

    The artifact is treated as production configuration. This loader
    does not execute calibration experiments.
    """
    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"likelihood artifact not found: {artifact_path}"
        )

    if not artifact_path.is_file():
        raise ValueError(
            f"likelihood artifact path is not a file: {artifact_path}"
        )

    try:
        payload = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid likelihood artifact JSON: {artifact_path}"
        ) from exc

    _validate_artifact(payload)

    registry = LikelihoodModelRegistry()

    for item in payload["models"]:
        model = GaussianLikelihoodModel(
            mean_true=item["mean_true"],
            std_true=item["std_true"],
            mean_false=item["mean_false"],
            std_false=item["std_false"],
            measurement=item["measurement"],
        )

        registry.register(
            evidence_group=item["evidence_group"],
            hypothesis=item["hypothesis"],
            model=model,
        )

    return registry


def _validate_artifact(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError(
            "likelihood artifact must contain a JSON object"
        )

    schema_version = payload.get("schema_version")

    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            "unsupported likelihood artifact schema version: "
            f"{schema_version!r}"
        )

    if "models" not in payload:
        raise ValueError(
            "likelihood artifact must contain 'models'"
        )

    models = payload["models"]

    if not isinstance(models, list):
        raise ValueError(
            "likelihood artifact 'models' must be a list"
        )

    model_count = payload.get("model_count")

    if model_count != len(models):
        raise ValueError(
            "likelihood artifact model_count does not match "
            "the number of models"
        )

    for index, item in enumerate(models):
        if not isinstance(item, dict):
            raise ValueError(
                f"likelihood model at index {index} must be an object"
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

        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                "likelihood model at index "
                f"{index} is missing fields: "
                f"{sorted(missing)}"
            )

        for field in (
            "evidence_group",
            "hypothesis",
            "measurement",
        ):
            value = item[field]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"likelihood model field '{field}' "
                    f"at index {index} must be a non-empty string"
                )