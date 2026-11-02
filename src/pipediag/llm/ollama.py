from __future__ import annotations

from typing import Any

from pipediag.diagnostics.explanation import ExplanationSpec

from .prompt import SYSTEM_PROMPT, build_explanation_prompt
from .validator import (
    validate_explanation_fidelity,
    validate_explanation_output,
)

class OllamaExplanationGenerator:
    """
    Generates natural-language diagnostic explanations using Ollama.

    Ollama is an optional dependency. This class is only intended to be
    used when the PipeDiag LLM extra has been installed and an Ollama
    server is available.
    """

    def __init__(
        self,
        *,
        model: str = "phi3",
        client: Any | None = None,
        max_output_length: int = 12000,
    ) -> None:
        if not model or not isinstance(model, str):
            raise ValueError(
                "model must be a non-empty string"
            )

        self.model = model
        self.max_output_length = max_output_length

        if client is None:
            try:
                import ollama
            except ImportError as exc:
                raise ImportError(
                    "Ollama support requires the optional LLM dependency. "
                    'Install it with: pip install "pipediag[llm]"'
                ) from exc

            client = ollama

        self.client = client


    def generate(
        self,
        explanation: ExplanationSpec,
    ) -> str:
        if explanation is None:
            raise ValueError(
                "explanation cannot be None"
            )

        prompt = build_explanation_prompt(explanation)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Allow one initial generation and one correction attempt.
        for attempt in range(2):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate explanation using Ollama "
                    f"model '{self.model}': {exc}"
                ) from exc

            content = self._extract_content(response)


            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "LLM explanation output cannot be empty"
                )


            try:
                validated = validate_explanation_output(
                    content,
                    max_length=self.max_output_length,
                )

                return validate_explanation_fidelity(
                    validated,
                    explanation,
                )

            except ValueError as exc:
                if attempt == 1:
                    # Do not trust an incomplete LLM report as a complete report.
                    # Add exact facts from PipeDiag's structured result.
                    facts = self._build_authoritative_facts(explanation)

                    if isinstance(content, str) and content.strip():
                        final_report = (
                            f"{content.strip()}\n\n"
                            f"{facts}"
                        )
                    else:
                        final_report = facts

                    validated = validate_explanation_fidelity(
                        final_report,
                        explanation,
                    )

                    return validate_explanation_output(
                        validated,
                        max_length=self.max_output_length,
                    )

                # Retry with the original structured facts and the
                # precise reason the first response was rejected.
                previous_content = (
                    content
                    if isinstance(content, str)
                    else repr(content)
                )

                correction_prompt = (
                    "Your previous report failed validation for this reason:\n"
                    f"{exc}\n\n"
                    "Rewrite the complete report. Treat the original "
                    "structured diagnostic explanation as authoritative. "
                    "Include the exact decimal values for uncertainty, "
                    "reporting threshold, every reported hypothesis posterior, "
                    "and every supplied evidence measurement. Do not round, "
                    "omit, or change these values. State only the hypotheses "
                    "that were actually reported. Preserve the supplied "
                    "recommendations and distinguish diagnostic support "
                    "from causal proof. Return only the corrected report.\n\n"
                    "Original structured diagnostic explanation:\n"
                    f"{prompt}"
                )

                messages = [
                    *messages,
                    {
                        "role": "assistant",
                        "content": previous_content,
                    },
                    {
                        "role": "user",
                        "content": correction_prompt,
                    },
                ]

        # Defensive fallback; the loop normally returns or raises.
        raise RuntimeError(
            "Ollama explanation generation ended unexpectedly."
        )




    @staticmethod
    def _build_authoritative_facts(
        explanation: ExplanationSpec,
    ) -> str:
        """Render exact diagnostic facts directly from PipeDiag data."""
        lines = [
            "Authoritative PipeDiag diagnostic facts",
            f"Reporting threshold: {explanation.threshold!r}",
            f"Diagnostic uncertainty: {explanation.uncertainty!r}",
            f"Inconclusive: {explanation.is_inconclusive}",
            f"Reported hypothesis count: {len(explanation.hypotheses)}",
            f"Overall assessment: {explanation.overall_assessment}",
        ]

        for hypothesis in explanation.hypotheses:
            lines.extend([
                "",
                f"Hypothesis: {hypothesis.hypothesis_id}",
                f"Title: {hypothesis.title}",
                f"Posterior probability: {hypothesis.posterior!r}",
                "Evidence:",
            ])

            for evidence in hypothesis.evidence:
                lines.append(
                    f"- Group: {evidence.evidence_group}; "
                    f"Metric: {evidence.metric}; "
                    f"Value: {evidence.value!r}; "
                    f"Interpretation: {evidence.interpretation}"
                )

            lines.append("Explanation points:")
            lines.extend(
                f"- {point}"
                for point in hypothesis.explanation_points
            )
            lines.append(
                f"Recommendation: {hypothesis.recommendation}"
            )

        return "\n".join(lines)



    @staticmethod
    def _extract_content(response: Any) -> str:
        """
        Extract assistant text from the Ollama response.

        Supports the normal Ollama Python client response shape and
        dictionary-like test doubles.
        """

        if response is None:
            raise ValueError(
                "Ollama returned no response"
            )

        if isinstance(response, dict):
            message = response.get("message")

            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    return content

        message = getattr(response, "message", None)

        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return content

        raise ValueError(
            "Ollama response did not contain assistant message content"
        )