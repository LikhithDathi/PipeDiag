from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.full_system_validation import (  # noqa: E402
    build_calibrated_likelihood_registry,
)


OUTPUT_PATHS = [
    PROJECT_ROOT / "artifacts" / "calibration" / "likelihood_models.json",
    PROJECT_ROOT / "src" / "pipediag" / "calibration" / "likelihood_models.json",
]

EXPECTED_MODEL_COUNT = 11
SCHEMA_VERSION = 1


def build_artifact() -> dict:
    registry = build_calibrated_likelihood_registry()

    models = []

    for (evidence_group, hypothesis), model in sorted(
        registry.models().items()
    ):
        models.append(
            {
                "evidence_group": evidence_group,
                "hypothesis": hypothesis,
                "measurement": model.measurement,
                "mean_true": model.mean_true,
                "std_true": model.std_true,
                "mean_false": model.mean_false,
                "std_false": model.std_false,
            }
        )

    if len(models) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_COUNT} calibration models, "
            f"but generated {len(models)}."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_count": len(models),
        "models": models,
    }


def export_calibration() -> None:
    artifact = build_artifact()

    serialized = json.dumps(
        artifact,
        indent=2,
        sort_keys=True,
    )

    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            serialized,
            encoding="utf-8",
        )

        # Verify the written JSON.
        written_artifact = json.loads(
            output_path.read_text(encoding="utf-8")
        )

        if written_artifact != artifact:
            raise ValueError(
                f"Calibration verification failed: {output_path}"
            )

        print(f"Verified: {output_path}")

    print(f"\nSuccessfully exported {len(artifact['models'])} models.")
    print("Both calibration artifacts are synchronized.")


if __name__ == "__main__":
    export_calibration()