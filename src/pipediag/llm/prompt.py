from __future__ import annotations

from pipediag.diagnostics.explanation import ExplanationSpec


SYSTEM_PROMPT = """
You are the explanation component of PipeDiag, an ML pipeline
diagnostic framework.

Your task is to convert the provided structured diagnostic explanation
into clear, technically accurate natural language for a developer.

The structured explanation is authoritative.

You MUST:
- preserve all posterior probabilities exactly;
- preserve the reported uncertainty;
- preserve the reporting threshold;
- preserve the reported hypotheses;
- state the number of reported hypotheses accurately;
- describe only hypotheses listed in the structured input as reported;
- never claim that an unreported hypothesis crossed the threshold;
- preserve the supplied evidence;
- preserve the supplied explanation points;
- preserve the supplied recommendations;
- clearly distinguish probabilistic diagnostic support from causal certainty;
- only explain information contained in the supplied structured explanation.

You MUST NOT:
- perform new diagnosis;
- add new evidence;
- invent measurements;
- change posterior probabilities;
- remove reported hypotheses;
- introduce hypotheses that are not supplied;
- change the inconclusive/threshold decision;
- claim that a hypothesis is a proven root cause;
- invent remediation steps or recommendations;
- infer information about the user's code that is not present in the input.

Return only the natural-language explanation.
""".strip()


def build_explanation_prompt(
    explanation: ExplanationSpec,
) -> str:
    """
    Build the user-facing prompt supplied to an LLM provider.

    The prompt contains only the structured ExplanationSpec and does not
    expose runtime objects, raw datasets, model objects, or diagnostic
    implementation details.
    """
    if explanation is None:
        raise ValueError(
            "explanation cannot be None"
        )

    sections: list[str] = []

    sections.append(
        "Structured diagnostic explanation:"
    )

    sections.append(
        f"Reporting threshold: {explanation.threshold:.6f}"
    )

    sections.append(
        f"Uncertainty: {explanation.uncertainty:.6f}"
    )

    sections.append(
        f"Inconclusive: {explanation.is_inconclusive}"
    )

    sections.append(
        f"Reported hypothesis count: {len(explanation.hypotheses)}"
    )

    sections.append(
        f"Overall assessment: {explanation.overall_assessment}"
    )

    sections.append(
        "Hypotheses:"
    )

    for hypothesis in explanation.hypotheses:
        sections.append(
            f"- ID: {hypothesis.hypothesis_id}"
        )
        sections.append(
            f"  Title: {hypothesis.title}"
        )
        sections.append(
            f"  Posterior: {hypothesis.posterior:.6f}"
        )

        sections.append(
            "  Evidence:"
        )

        for evidence in hypothesis.evidence:
            sections.append(
                f"    - Evidence group: {evidence.evidence_group}; "
                f"Metric: {evidence.metric}; "
                f"Value: {evidence.value}; "
                f"Interpretation: {evidence.interpretation}"
            )

        sections.append(
            "  Explanation points:"
        )

        for point in hypothesis.explanation_points:
            sections.append(
                f"    - {point}"
            )

        sections.append(
            f"  Recommendation: {hypothesis.recommendation}"
        )

    sections.append(
        "Report-writing requirements:"
    )
    sections.append(
        "- Begin with a short summary of the reported diagnosis."
    )
    sections.append(
        "- Explain how the supplied measurements support the reported hypothesis."
    )
    sections.append(
        "- Explain uncertainty and distinguish evidence from causal proof."
    )
    sections.append(
        "- Give the supplied recommendation without inventing additional fixes."
    )
    sections.append(
        "- Mention only the number of reported hypotheses stated above."
    )
    sections.append(
        "- Do not claim that any other hypothesis crossed the threshold."
    )
    sections.append(
        "- Keep probabilities, uncertainty, threshold, and measurements faithful "
        "to the structured input."
    )
    sections.append(
        "Produce a concise, technically accurate developer-facing report "
        "using clear language and short paragraphs."
    )

    return "\n".join(sections)