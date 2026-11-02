from __future__ import annotations

import math
import re

from pipediag.diagnostics.explanation import ExplanationSpec


DEFAULT_MAX_OUTPUT_LENGTH = 12000

_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"\s*%?"
)


def validate_explanation_output(
    output: str,
    *,
    max_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
) -> str:
    """
    Validate and normalize the presentation output.
    """
    if not isinstance(output, str):
        raise ValueError(
            "LLM explanation output must be a string"
        )

    normalized = output.strip()

    if not normalized:
        raise ValueError(
            "LLM explanation output cannot be empty"
        )

    if len(normalized) > max_length:
        raise ValueError(
            "LLM explanation output exceeds the maximum allowed length"
        )

    return normalized


def _extract_numeric_values(output: str) -> list[float]:
    """
    Extract numbers from prose.

    Percentages are converted to fractions, so 81.35% and
    0.8135 are treated as equivalent representations.
    """
    values: list[float] = []

    for match in _NUMBER_PATTERN.finditer(output):
        token = match.group().strip()
        is_percentage = token.endswith("%")

        if is_percentage:
            token = token[:-1].strip()

        try:
            value = float(token)
        except ValueError:
            continue

        if not math.isfinite(value):
            continue

        if is_percentage:
            value /= 100.0

        values.append(value)

    return values


def validate_explanation_fidelity(
    output: str,
    explanation: ExplanationSpec,
) -> str:
    """
    Verify that a generated report contains the key facts supplied
    by the authoritative ExplanationSpec.

    This is a presence check, not a complete semantic fact checker.
    """
    normalized = validate_explanation_output(output)

    if explanation is None:
        raise ValueError(
            "explanation cannot be None"
        )

    numeric_values = _extract_numeric_values(normalized)

    # Check uncertainty first to give a clear error for incomplete
    # reports, including reports for inconclusive diagnoses.
    required_values: list[tuple[str, float]] = [
        ("uncertainty", explanation.uncertainty),
        ("reporting threshold", explanation.threshold),
    ]

    for hypothesis in explanation.hypotheses:
        required_values.append(
            (
                f"posterior for {hypothesis.hypothesis_id}",
                hypothesis.posterior,
            )
        )

        for evidence in hypothesis.evidence:
            required_values.append(
                (
                    f"evidence value for {hypothesis.hypothesis_id} "
                    f"({evidence.metric})",
                    float(evidence.value),
                )
            )

    for label, expected in required_values:
        if not any(
            math.isclose(
                actual,
                expected,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for actual in numeric_values
        ):
            raise ValueError(
                f"LLM explanation is missing or changes required "
                f"{label}: {expected}"
            )

    for hypothesis in explanation.hypotheses:
        if hypothesis.hypothesis_id not in normalized:
            raise ValueError(
                "LLM explanation is missing reported hypothesis "
                f"{hypothesis.hypothesis_id}"
            )

    return normalized
