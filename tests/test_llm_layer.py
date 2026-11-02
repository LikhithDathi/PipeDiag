import pytest

from pipediag.llm.generator import (
    DeterministicExplanationGenerator,
)
from pipediag.llm.prompt import (
    SYSTEM_PROMPT,
    build_explanation_prompt,
)
from pipediag.llm.validator import (
    validate_explanation_output,
)


def test_deterministic_generator_requires_explanation():
    generator = DeterministicExplanationGenerator()

    with pytest.raises(
        ValueError,
        match="explanation cannot be None",
    ):
        generator.generate(None)


def test_deterministic_generator_returns_report():
    from pipediag.diagnostics.explanation import ExplanationSpec

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment="No sufficiently supported diagnosis.",
    )

    generator = DeterministicExplanationGenerator()

    result = generator.generate(explanation)

    assert isinstance(result, str)
    assert result.strip()


def test_prompt_requires_explanation():
    with pytest.raises(
        ValueError,
        match="explanation cannot be None",
    ):
        build_explanation_prompt(None)


def test_prompt_contains_structured_information():
    from pipediag.diagnostics.explanation import ExplanationSpec

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment="No sufficiently supported diagnosis.",
    )

    prompt = build_explanation_prompt(explanation)

    assert "Reporting threshold:" in prompt
    assert "Uncertainty:" in prompt
    assert "Inconclusive:" in prompt
    assert "Overall assessment:" in prompt
    assert "Hypotheses:" in prompt


def test_system_prompt_contains_llm_constraints():
    assert "MUST NOT" in SYSTEM_PROMPT
    assert "add new evidence" in SYSTEM_PROMPT
    assert "change posterior probabilities" in SYSTEM_PROMPT
    assert "perform new diagnosis" in SYSTEM_PROMPT


def test_validator_accepts_valid_output():
    result = validate_explanation_output(
        "The model shows evidence of a generalization problem."
    )

    assert result == (
        "The model shows evidence of a generalization problem."
    )


def test_validator_rejects_empty_output():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        validate_explanation_output("   ")


def test_validator_rejects_non_string_output():
    with pytest.raises(
        ValueError,
        match="must be a string",
    ):
        validate_explanation_output(None)


def test_validator_rejects_excessively_long_output():
    with pytest.raises(
        ValueError,
        match="maximum allowed length",
    ):
        validate_explanation_output(
            "x" * 12001
        )





def test_ollama_generator_sends_structured_prompt():
    from pipediag.diagnostics.explanation import ExplanationSpec
    from pipediag.llm.ollama import OllamaExplanationGenerator

    expected = (
        "The diagnostic result indicates limited evidence. "
        "Uncertainty is 0.10 and reporting threshold is 0.20."
    )

    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "message": {
                    "content": expected,
                }
            }

    client = FakeClient()

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment="No sufficiently supported diagnosis.",
    )

    generator = OllamaExplanationGenerator(
        model="phi3",
        client=client,
    )

    result = generator.generate(explanation)

    assert result == expected
    assert len(client.calls) == 1

    call = client.calls[0]

    assert call["model"] == "phi3"
    assert len(call["messages"]) == 2
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"

    assert "Reporting threshold:" in call["messages"][1]["content"]
    assert "Uncertainty:" in call["messages"][1]["content"]




def test_ollama_generator_supports_response_objects():
    from pipediag.diagnostics.explanation import ExplanationSpec
    from pipediag.llm.ollama import OllamaExplanationGenerator

    expected = (
        "Explanation generated successfully. "
        "Uncertainty is 0.10 and reporting threshold is 0.20."
    )

    class Message:
        content = expected

    class Response:
        message = Message()

    class FakeClient:
        def chat(self, **kwargs):
            return Response()

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment="No sufficiently supported diagnosis.",
    )

    generator = OllamaExplanationGenerator(
        model="phi3",
        client=FakeClient(),
    )

    result = generator.generate(explanation)

    assert result == expected



def test_ollama_generator_rejects_empty_response():
    from pipediag.diagnostics.explanation import ExplanationSpec
    from pipediag.llm.ollama import OllamaExplanationGenerator

    class FakeClient:
        def chat(self, **kwargs):
            return {
                "message": {
                    "content": "   "
                }
            }

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment="No sufficiently supported diagnosis.",
    )

    generator = OllamaExplanationGenerator(
        model="phi3",
        client=FakeClient(),
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        generator.generate(explanation)


def test_ollama_generator_wraps_client_failure():
    from pipediag.diagnostics.explanation import ExplanationSpec
    from pipediag.llm.ollama import OllamaExplanationGenerator

    class FakeClient:
        def chat(self, **kwargs):
            raise RuntimeError("Ollama server unavailable")

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment="No sufficiently supported diagnosis.",
    )

    generator = OllamaExplanationGenerator(
        model="phi3",
        client=FakeClient(),
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to generate explanation using Ollama",
    ):
        generator.generate(explanation)






def test_ollama_real_generation():
    from pipediag.diagnostics.explanation import ExplanationSpec
    from pipediag.llm.ollama import OllamaExplanationGenerator

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment=(
            "No sufficiently supported diagnosis was identified."
        ),
    )

    generator = OllamaExplanationGenerator(
        model="phi3:latest",
    )

    result = generator.generate(explanation)

    assert isinstance(result, str)
    assert result.strip()








def test_report_fidelity_requires_all_reported_hypothesis_ids():
    from pipediag.diagnostics.explanation import (
        EvidenceExplanation,
        ExplanationSpec,
        HypothesisExplanation,
    )
    from pipediag.llm.validator import validate_explanation_fidelity

    hypothesis = HypothesisExplanation(
        hypothesis_id="H2",
        title="Generalization Failure",
        posterior=0.8135,
        evidence=(
            EvidenceExplanation(
                evidence_group="G2",
                metric="generalization_gap",
                value=0.14,
                interpretation="Training performance exceeds evaluation performance.",
            ),
        ),
        explanation_points=("A performance gap was observed.",),
        recommendation="Compare training and evaluation performance.",
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.3841,
        hypotheses=(hypothesis,),
        overall_assessment="Generalization failure is the leading hypothesis.",
    )

    report = (
        "H2 Generalization Failure has posterior 0.8135. "
        "The generalization gap is 0.14. "
        "Uncertainty is 0.3841 and the reporting threshold is 0.20."
    )

    validate_explanation_fidelity(report, explanation)


def test_report_fidelity_rejects_missing_posterior():
    import pytest
    from pipediag.diagnostics.explanation import ExplanationSpec
    from pipediag.llm.validator import validate_explanation_fidelity

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment="No sufficiently supported diagnosis.",
    )

    with pytest.raises(ValueError, match="uncertainty"):
        validate_explanation_fidelity(
            "No diagnosis is reported.",
            explanation,
        )






def test_report_fidelity_accepts_exact_facts_and_percentages():
    from pipediag.diagnostics.explanation import (
        EvidenceExplanation,
        ExplanationSpec,
        HypothesisExplanation,
    )
    from pipediag.llm.validator import validate_explanation_fidelity

    hypothesis = HypothesisExplanation(
        hypothesis_id="H2",
        title="Generalization Failure",
        posterior=0.8135,
        evidence=(
            EvidenceExplanation(
                evidence_group="G2",
                metric="generalization_gap",
                value=0.14,
                interpretation="Training performance exceeds evaluation performance.",
            ),
        ),
        explanation_points=("A performance gap was observed.",),
        recommendation="Compare training and evaluation performance.",
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.3841,
        hypotheses=(hypothesis,),
        overall_assessment="Generalization failure is the leading hypothesis.",
    )

    report = (
        "H2 Generalization Failure has posterior 81.35%. "
        "The generalization gap is 14%. "
        "Diagnostic uncertainty is 0.3841. "
        "The reporting threshold is 20%."
    )

    assert validate_explanation_fidelity(report, explanation) == report


def test_report_fidelity_rejects_rounded_posterior():
    import pytest
    from pipediag.diagnostics.explanation import (
        EvidenceExplanation,
        ExplanationSpec,
        HypothesisExplanation,
    )
    from pipediag.llm.validator import validate_explanation_fidelity

    hypothesis = HypothesisExplanation(
        hypothesis_id="H2",
        title="Generalization Failure",
        posterior=0.8135,
        evidence=(
            EvidenceExplanation(
                evidence_group="G2",
                metric="generalization_gap",
                value=0.14,
                interpretation="Training performance exceeds evaluation performance.",
            ),
        ),
        explanation_points=("A performance gap was observed.",),
        recommendation="Compare training and evaluation performance.",
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.3841,
        hypotheses=(hypothesis,),
        overall_assessment="Generalization failure is the leading hypothesis.",
    )

    report = (
        "H2 Generalization Failure has posterior 0.81. "
        "The generalization gap is 0.14. "
        "Uncertainty is 0.3841. "
        "The reporting threshold is 0.20."
    )

    with pytest.raises(ValueError, match="posterior for H2"):
        validate_explanation_fidelity(report, explanation)






def test_ollama_generator_retries_when_fidelity_validation_fails():
    from pipediag.diagnostics.explanation import ExplanationSpec
    from pipediag.llm.ollama import OllamaExplanationGenerator

    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)

            if len(self.calls) == 1:
                content = "The evidence is inconclusive."

            else:
                content = (
                    "The evidence is inconclusive. "
                    "Diagnostic uncertainty is 0.10 and "
                    "the reporting threshold is 0.20."
                )

            return {"message": {"content": content}}

    client = FakeClient()

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.10,
        hypotheses=(),
        overall_assessment="No sufficiently supported diagnosis.",
    )

    generator = OllamaExplanationGenerator(
        model="phi3",
        client=client,
    )

    result = generator.generate(explanation)

    assert len(client.calls) == 2
    assert "uncertainty is 0.10" in result
    assert "threshold is 0.20" in result
    assert "failed validation" in client.calls[1]["messages"][-1]["content"]
