from pipediag.application import DiagnosticApplication
from pipediag.llm.ollama import OllamaExplanationGenerator
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
import pipediag


class FakeEvidenceProvider:
    def collect(self, source):
        return []


class FakeDiagnosticEngine:
    def diagnose(self, evidence_groups, metadata=None):
        return DiagnosticResult(
            posterior={
                "H1": 1 / 9,
                "H2": 1 / 9,
                "H3": 1 / 9,
                "H4": 1 / 9,
                "H5": 1 / 9,
                "H6": 1 / 9,
                "H7": 1 / 9,
                "H8": 1 / 9,
                "H9": 1 / 9,
            },
            uncertainty=1.0,
            evidence=[],
            metadata=metadata or {},
        )


class FakeOllamaClient:
    def chat(self, **kwargs):
        return {
            "message": {
                "content": (
                    " The diagnostic evidence does not support a sufficiently confident "
                    "conclusion. Diagnostic uncertainty is 1.0 and the reporting threshold "
                    "is 0.20."
                )
            }
        }


def test_public_run_supports_ollama_explanation():
    generator = OllamaExplanationGenerator(
        model="phi3:latest",
        client=FakeOllamaClient(),
    )

    application = DiagnosticApplication(
        evidence_provider=FakeEvidenceProvider(),
        diagnostic_engine=FakeDiagnosticEngine(),
        explanation_generator=generator,
    )

    with pipediag.start(application=application) as run:
        result = run.result()

    assert result.report == (
        "The diagnostic evidence does not support "
        "a sufficiently confident conclusion. "
        "Diagnostic uncertainty is 1.0 and the reporting threshold "
        "is 0.20."
    )

    assert result.explanation is not None
    assert result.diagnostic_result is not None