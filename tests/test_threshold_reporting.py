import math

import pytest

from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.threshold_reporting import (
    DEFAULT_REPORTING_THRESHOLD,
    ReportedHypothesis,
    ThresholdDiagnosticReport,
    ThresholdReporter,
)


HYPOTHESES = (
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H8",
    "H9",
)


def _diagnostic_result(
    posterior: dict[str, float],
    uncertainty: float = 0.4,
) -> DiagnosticResult:
    return DiagnosticResult(
        posterior=posterior,
        uncertainty=uncertainty,
        evidence=(),
        metadata={},
    )


def test_default_reporting_threshold_is_0_20():
    assert DEFAULT_REPORTING_THRESHOLD == 0.20


def test_reported_hypothesis_validates_id():
    with pytest.raises(ValueError):
        ReportedHypothesis(
            hypothesis_id="",
            posterior=0.5,
        )


def test_reported_hypothesis_validates_posterior():
    with pytest.raises(ValueError):
        ReportedHypothesis(
            hypothesis_id="H1",
            posterior=1.1,
        )


def test_report_returns_hypotheses_above_threshold():
    result = _diagnostic_result(
        {
            "H1": 0.10,
            "H2": 0.25,
            "H3": 0.55,
            "H4": 0.10,
        }
    )

    report = ThresholdReporter(
        threshold=0.20
    ).report(result)

    assert report.reported_hypothesis_ids == (
        "H3",
        "H2",
    )


def test_hypotheses_are_ranked_by_posterior():
    result = _diagnostic_result(
        {
            "H1": 0.10,
            "H2": 0.60,
            "H3": 0.20,
            "H4": 0.10,
        }
    )

    report = ThresholdReporter(
        threshold=0.20
    ).report(result)

    assert report.reported_hypothesis_ids == (
        "H2",
        "H3",
    )


def test_threshold_is_inclusive():
    result = _diagnostic_result(
        {
            "H1": 0.20,
            "H2": 0.19,
            "H3": 0.61,
        }
    )

    report = ThresholdReporter(
        threshold=0.20
    ).report(result)

    assert report.reported_hypothesis_ids == (
        "H3",
        "H1",
    )


def test_no_hypothesis_above_threshold_is_inconclusive():
    posterior = {
        hypothesis: 1.0 / 9.0
        for hypothesis in HYPOTHESES
    }

    result = _diagnostic_result(
        posterior,
        uncertainty=1.0,
    )

    report = ThresholdReporter(
        threshold=0.20
    ).report(result)

    assert report.is_inconclusive
    assert report.reported_hypotheses == ()


def test_multiple_hypotheses_can_be_reported():
    result = _diagnostic_result(
        {
            "H1": 0.05,
            "H2": 0.45,
            "H3": 0.35,
            "H4": 0.15,
        }
    )

    report = ThresholdReporter(
        threshold=0.20
    ).report(result)

    assert report.reported_hypothesis_ids == (
        "H2",
        "H3",
    )


def test_uncertainty_is_preserved():
    result = _diagnostic_result(
        {
            "H1": 1.0,
        },
        uncertainty=0.123,
    )

    report = ThresholdReporter(
        threshold=0.20
    ).report(result)

    assert math.isclose(
        report.uncertainty,
        0.123,
    )


def test_custom_threshold_is_supported():
    result = _diagnostic_result(
        {
            "H1": 0.40,
            "H2": 0.30,
            "H3": 0.10,
            "H4": 0.20,
        }
    )

    report = ThresholdReporter(
        threshold=0.35
    ).report(result)

    assert report.reported_hypothesis_ids == (
        "H1",
    )


def test_threshold_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        ThresholdReporter(threshold=-0.1)

    with pytest.raises(ValueError):
        ThresholdReporter(threshold=1.1)


def test_none_result_is_rejected():
    reporter = ThresholdReporter()

    with pytest.raises(ValueError):
        reporter.report(None)