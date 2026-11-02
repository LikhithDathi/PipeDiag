import pytest

from pipediag.application import (
    DiagnosticApplication,
    DiagnosticRunResult,
)
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.explanation import (
    ExplanationBuilder,
    ExplanationSpec,
)
from pipediag.diagnostics.report import DiagnosticReportRenderer


HYPOTHESES = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H8",
    "H9",
]


def _diagnostic_result():
    posterior = {
        "H1": 0.05,
        "H2": 0.05,
        "H3": 0.70,
        "H4": 0.05,
        "H5": 0.05,
        "H6": 0.02,
        "H7": 0.03,
        "H8": 0.03,
        "H9": 0.02,
    }

    evidence = [
        EvidenceGroup(
            name="G3",
            value=0.63,
            status="observed",
            metadata={
                "metric": "mean_absolute_standardized_shift"
            },
        )
    ]

    return DiagnosticResult(
        posterior=posterior,
        uncertainty=0.40,
        evidence=evidence,
    )


class FakeEvidenceProvider:
    def __init__(self):
        self.sources = []

    def collect(self, source):
        self.sources.append(source)

        return [
            EvidenceGroup(
                name="G3",
                value=0.63,
                status="observed",
                metadata={
                    "metric": "mean_absolute_standardized_shift"
                },
            )
        ]


class FakeDiagnosticEngine:
    def __init__(self):
        self.calls = []

    def diagnose(
        self,
        evidence_groups,
        metadata=None,
    ):
        self.calls.append(
            {
                "evidence_groups": evidence_groups,
                "metadata": metadata,
            }
        )

        return _diagnostic_result()


class FakeExplanationBuilder:
    def __init__(self):
        self.calls = []

    def build(self, diagnostic_result):
        self.calls.append(diagnostic_result)

        return ExplanationSpec(
            threshold=0.20,
            uncertainty=diagnostic_result.uncertainty,
            hypotheses=(),
            overall_assessment="Test explanation",
        )


class FakeReportRenderer:
    def __init__(self):
        self.calls = []

    def render(self, explanation):
        self.calls.append(explanation)

        return "TEST REPORT"


def test_run_returns_complete_result():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    result = application.run("pipeline")

    assert isinstance(
        result,
        DiagnosticRunResult,
    )

    assert isinstance(
        result.diagnostic_result,
        DiagnosticResult,
    )

    assert isinstance(
        result.explanation,
        ExplanationSpec,
    )

    assert isinstance(
        result.report,
        str,
    )


def test_run_passes_source_to_evidence_provider():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    source = object()

    application.run(source)

    assert provider.sources == [source]


def test_run_passes_evidence_to_diagnostic_engine():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    application.run("pipeline")

    assert len(engine.calls) == 1

    evidence = engine.calls[0]["evidence_groups"]

    assert len(evidence) == 1
    assert evidence[0].name == "G3"


def test_run_passes_metadata_to_diagnostic_engine():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    metadata = {
        "pipeline_name": "example",
        "run_id": "run-1",
    }

    application.run(
        "pipeline",
        metadata=metadata,
    )

    assert engine.calls[0]["metadata"] == metadata


def test_run_connects_explanation_builder():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()
    explanation_builder = FakeExplanationBuilder()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
        explanation_builder=explanation_builder,
    )

    result = application.run("pipeline")

    assert len(explanation_builder.calls) == 1

    assert (
        explanation_builder.calls[0]
        is result.diagnostic_result
    )


def test_run_connects_report_renderer():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()
    explanation_builder = FakeExplanationBuilder()
    renderer = FakeReportRenderer()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
        explanation_builder=explanation_builder,
        report_renderer=renderer,
    )

    result = application.run("pipeline")

    assert len(renderer.calls) == 1

    assert (
        renderer.calls[0]
        is result.explanation
    )

    assert result.report == "TEST REPORT"


def test_default_explanation_builder_is_created():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    assert isinstance(
        application.explanation_builder,
        ExplanationBuilder,
    )


def test_default_report_renderer_is_created():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    assert isinstance(
        application.report_renderer,
        DiagnosticReportRenderer,
    )


def test_none_evidence_provider_is_rejected():
    engine = FakeDiagnosticEngine()

    with pytest.raises(
        ValueError,
        match="evidence_provider cannot be None",
    ):
        DiagnosticApplication(
            evidence_provider=None,
            diagnostic_engine=engine,
        )


def test_none_diagnostic_engine_is_rejected():
    provider = FakeEvidenceProvider()

    with pytest.raises(
        ValueError,
        match="diagnostic_engine cannot be None",
    ):
        DiagnosticApplication(
            evidence_provider=provider,
            diagnostic_engine=None,
        )


def test_none_evidence_is_rejected():
    class NoneEvidenceProvider:
        def collect(self, source):
            return None

    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=NoneEvidenceProvider(),
        diagnostic_engine=engine,
    )

    with pytest.raises(
        ValueError,
        match="returned None",
    ):
        application.run("pipeline")


def test_result_preserves_all_layers():
    provider = FakeEvidenceProvider()
    engine = FakeDiagnosticEngine()

    application = DiagnosticApplication(
        evidence_provider=provider,
        diagnostic_engine=engine,
    )

    result = application.run("pipeline")

    assert result.diagnostic_result is not None
    assert result.explanation is not None
    assert result.report


def test_diagnostic_run_result_requires_diagnostic_result():
    builder = ExplanationBuilder()

    explanation = builder.build(
        _diagnostic_result()
    )

    with pytest.raises(
        ValueError,
        match="diagnostic_result",
    ):
        DiagnosticRunResult(
            diagnostic_result=None,
            explanation=explanation,
            report="report",
        )


def test_diagnostic_run_result_requires_explanation():
    diagnostic_result = _diagnostic_result()

    with pytest.raises(
        ValueError,
        match="explanation",
    ):
        DiagnosticRunResult(
            diagnostic_result=diagnostic_result,
            explanation=None,
            report="report",
        )


def test_diagnostic_run_result_requires_string_report():
    diagnostic_result = _diagnostic_result()

    builder = ExplanationBuilder()

    explanation = builder.build(
        diagnostic_result
    )

    with pytest.raises(
        ValueError,
        match="report must be a string",
    ):
        DiagnosticRunResult(
            diagnostic_result=diagnostic_result,
            explanation=explanation,
            report=None,
        )