from __future__ import annotations

import json
from pathlib import Path

from experiments.full_system_validation import (
    build_calibrated_likelihood_registry,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATHS = [
    ROOT / "artifacts" / "calibration" / "likelihood_models.json",
    ROOT / "src" / "pipediag" / "calibration" / "likelihood_models.json",
]


def main() -> None:
    print("Building calibrated likelihood registry...")

    registry = build_calibrated_likelihood_registry()
    registered_models = registry.models()

    models = []


    for key, model in registered_models.items():
        if len(key) == 3:
            evidence_group, hypothesis, measurement = key
        elif len(key) == 2:
            evidence_group, hypothesis = key
            measurement = model.measurement
        else:
            raise ValueError(
                f"Unexpected likelihood registry key: {key}"
            )

        models.append({
            "evidence_group": evidence_group,
            "hypothesis": hypothesis,
            "measurement": measurement,
            "mean_true": float(model.mean_true),
            "std_true": float(model.std_true),
            "mean_false": float(model.mean_false),
            "std_false": float(model.std_false),
        })


    runtime_key = (
        "G1",
        "H1",
        "runtime_contamination_rate",
    )

    if runtime_key not in registered_models:
        raise RuntimeError(
            "Runtime G1/H1 calibration model was not generated."
        )

    if len(models) != 11:
        raise RuntimeError(
            f"Expected 11 calibrated models, found {len(models)}."
        )

    payload = {
        "schema_version": 1,
        "model_count": len(models),
        "models": models,
    }

    serialized = json.dumps(payload, indent=2)

    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            serialized + "\n",
            encoding="utf-8",
        )
        print(f"Saved: {output_path}")

    print(f"\nSuccessfully generated {len(models)} models.")
    print("Runtime model: G1 -> H1 -> runtime_contamination_rate")


if __name__ == "__main__":
    main()