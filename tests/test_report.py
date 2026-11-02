import pytest

from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.explanation import ExplanationBuilder
from pipediag.diagnostics.explanation import ExplanationSpec
from pipediag.diagnostics.report import DiagnosticReportRenderer


def _result(
    *,
    posterior=None,
    evidence=(),
    uncertainty=0.4,
):
    if posterior is None:
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

    return DiagnosticResult(
        posterior=posterior,
        uncertainty=uncertainty,
        evidence=list(evidence),
    )


def test_renderer_returns_string():
    builder = ExplanationBuilder()
    explanation = builder.build(_result())

    renderer = DiagnosticReportRenderer()

    report = renderer.render(explanation)

    assert isinstance(report, str)
    assert report


def test_renderer_contains_report_title():
    builder = ExplanationBuilder()
    explanation = builder.build(_result())

    report = DiagnosticReportRenderer().render(explanation)

    assert "# Diagnostic Report" in report


def test_renderer_contains_threshold():
    builder = ExplanationBuilder(
        threshold=0.20
    )

    explanation = builder.build(_result())

    report = DiagnosticReportRenderer().render(explanation)

    assert "**Posterior reporting threshold:** 0.20" in report


def test_renderer_contains_uncertainty():
    builder = ExplanationBuilder()

    explanation = builder.build(
        _result(
            uncertainty=0.73
        )
    )

    report = DiagnosticReportRenderer().render(explanation)

    assert "**Uncertainty:** 0.73" in report


def test_renderer_contains_reported_hypothesis():
    builder = ExplanationBuilder()

    explanation = builder.build(_result())

    report = DiagnosticReportRenderer().render(explanation)

    assert "Distribution Shift (H3)" in report


def test_renderer_contains_posterior_support():
    builder = ExplanationBuilder()

    explanation = builder.build(_result())

    report = DiagnosticReportRenderer().render(explanation)

    assert "**Posterior support:** 0.70" in report


def test_renderer_contains_evidence():
    evidence = EvidenceGroup(
        name="G3",
        value=0.63,
        status="observed",
        metadata={
            "metric": "mean_absolute_standardized_shift"
        },
    )

    builder = ExplanationBuilder()

    explanation = builder.build(
        _result(
            evidence=[evidence]
        )
    )

    report = DiagnosticReportRenderer().render(explanation)

    assert "mean_absolute_standardized_shift" in report
    assert "0.630" in report
    assert "substantial" in report


def test_renderer_contains_recommendation():
    builder = ExplanationBuilder()

    explanation = builder.build(_result())

    report = DiagnosticReportRenderer().render(explanation)

    assert "Recommended investigation" in report
    assert (
        "Compare reference and evaluation feature distributions"
        in report
    )


def test_renderer_contains_causal_limitation():
    builder = ExplanationBuilder()

    explanation = builder.build(_result())

    report = DiagnosticReportRenderer().render(explanation)

    assert "do not establish a causal root cause" in report


def test_renderer_reports_multiple_hypotheses():
    posterior = {
        "H1": 0.30,
        "H2": 0.05,
        "H3": 0.40,
        "H4": 0.15,
        "H5": 0.02,
        "H6": 0.01,
        "H7": 0.02,
        "H8": 0.01,
        "H9": 0.04,
    }

    builder = ExplanationBuilder(
        threshold=0.20
    )

    explanation = builder.build(
        _result(
            posterior=posterior
        )
    )

    report = DiagnosticReportRenderer().render(explanation)

    assert "1. Distribution Shift (H3)" in report
    assert "2. Data Leakage (H1)" in report


def test_renderer_handles_inconclusive_result():
    posterior = {
        hypothesis_id: 1.0 / 9.0
        for hypothesis_id in [
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
    }

    builder = ExplanationBuilder()

    explanation = builder.build(
        _result(
            posterior=posterior,
            uncertainty=1.0,
        )
    )

    assert explanation.is_inconclusive

    report = DiagnosticReportRenderer().render(explanation)

    assert (
        "No hypothesis reached the posterior reporting threshold"
        in report
    )
    assert (
        "does not support a dominant diagnostic explanation"
        in report
    )


def test_renderer_does_not_add_unreported_hypotheses():
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

    builder = ExplanationBuilder()

    explanation = builder.build(
        _result(
            posterior=posterior
        )
    )

    report = DiagnosticReportRenderer().render(explanation)

    assert "Distribution Shift (H3)" in report
    assert "Data Leakage (H1)" not in report
    assert "Class Imbalance (H4)" not in report


def test_renderer_rejects_none():
    renderer = DiagnosticReportRenderer()

    with pytest.raises(ValueError, match="cannot be None"):
        renderer.render(None)


def test_renderer_formats_zero_value():
    posterior = {
        "H1": 0.05,
        "H2": 0.05,
        "H3": 0.05,
        "H4": 0.05,
        "H5": 0.70,
        "H6": 0.02,
        "H7": 0.03,
        "H8": 0.03,
        "H9": 0.02,
    }

    evidence = EvidenceGroup(
        name="G5",
        value=0.0,
        status="observed",
        metadata={
            "metric": "missing_value_rate"
        },
    )

    builder = ExplanationBuilder()

    explanation = builder.build(
        _result(
            posterior=posterior,
            evidence=[evidence],
        )
    )

    report = DiagnosticReportRenderer().render(explanation)

    assert "**missing_value_rate:** 0 —" in report


def test_renderer_preserves_evidence_value():
    evidence = EvidenceGroup(
        name="G3",
        value=1.2345,
        status="observed",
        metadata={
            "metric": "mean_absolute_standardized_shift"
        },
    )

    posterior = {
        "H1": 0.02,
        "H2": 0.02,
        "H3": 0.82,
        "H4": 0.02,
        "H5": 0.02,
        "H6": 0.02,
        "H7": 0.03,
        "H8": 0.03,
        "H9": 0.02,
    }

    builder = ExplanationBuilder()

    explanation = builder.build(
        _result(
            posterior=posterior,
            evidence=[evidence],
        )
    )

    report = DiagnosticReportRenderer().render(explanation)

    assert "1.23" in report


def test_renderer_accepts_explanation_spec_directly():
    builder = ExplanationBuilder()
    explanation = builder.build(_result())

    assert isinstance(explanation, ExplanationSpec)

    renderer = DiagnosticReportRenderer()

    report = renderer.render(explanation)

    assert report.startswith("# Diagnostic Report")