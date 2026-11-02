from __future__ import annotations

from pipediag.diagnostics.explanation import ExplanationSpec


class DiagnosticReportRenderer:
    """
    Renders an ExplanationSpec as a deterministic Markdown report.

    This class performs no inference, evidence interpretation, thresholding,
    or causal reasoning. It only converts the structured explanation
    specification into human-readable output.
    """

    def render(self, explanation: ExplanationSpec) -> str:
        if explanation is None:
            raise ValueError(
                "explanation cannot be None"
            )

        lines: list[str] = []

        lines.append("# Diagnostic Report")
        lines.append("")

        lines.append(
            f"**Posterior reporting threshold:** "
            f"{explanation.threshold:.2f}"
        )
        lines.append(
            f"**Uncertainty:** "
            f"{explanation.uncertainty:.2f}"
        )
        lines.append("")

        lines.append(
            f"**Assessment:** {explanation.overall_assessment}"
        )
        lines.append("")

        if explanation.is_inconclusive:
            lines.extend(
                self._render_inconclusive(
                    explanation
                )
            )
        else:
            lines.extend(
                self._render_hypotheses(
                    explanation
                )
            )

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            "**Important:** Posterior values represent probabilistic "
            "diagnostic support. They do not establish a causal root cause."
        )

        return "\n".join(lines)

    def _render_inconclusive(
        self,
        explanation: ExplanationSpec,
    ) -> list[str]:
        lines: list[str] = []

        lines.append("## Diagnostic Conclusion")
        lines.append("")
        lines.append(
            "No hypothesis reached the posterior reporting threshold."
        )
        lines.append(
            "The available evidence does not support a dominant "
            "diagnostic explanation."
        )
        lines.append("")

        lines.append(
            "The observed evidence should still be reviewed when "
            "investigating the pipeline."
        )

        return lines

    def _render_hypotheses(
        self,
        explanation: ExplanationSpec,
    ) -> list[str]:
        lines: list[str] = []

        lines.append("## Potential Explanations")
        lines.append("")

        for index, hypothesis in enumerate(
            explanation.hypotheses,
            start=1,
        ):
            lines.extend(
                self._render_hypothesis(
                    index=index,
                    hypothesis=hypothesis,
                )
            )

        return lines

    def _render_hypothesis(
        self,
        *,
        index: int,
        hypothesis,
    ) -> list[str]:
        lines: list[str] = []

        lines.append(
            f"### {index}. {hypothesis.title} "
            f"({hypothesis.hypothesis_id})"
        )
        lines.append("")

        lines.append(
            f"**Posterior support:** "
            f"{hypothesis.posterior:.2f}"
        )
        lines.append("")

        lines.append("**Why it was reported:**")
        lines.append("")

        for point in hypothesis.explanation_points:
            lines.append(
                f"- {point}"
            )

        lines.append("")

        if hypothesis.evidence:
            lines.append("**Supporting evidence:**")
            lines.append("")

            for evidence in hypothesis.evidence:
                lines.append(
                    f"- **{evidence.metric}:** "
                    f"{self._format_value(evidence.value)} — "
                    f"{evidence.interpretation}"
                )

            lines.append("")

        lines.append("**Recommended investigation:**")
        lines.append("")
        lines.append(
            f"- {hypothesis.recommendation}"
        )
        lines.append("")

        return lines

    @staticmethod
    def _format_value(value: float) -> str:
        if value == 0.0:
            return "0"

        if abs(value) >= 1.0:
            return f"{value:.2f}"

        return f"{value:.3f}"