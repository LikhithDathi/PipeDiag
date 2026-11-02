import pytest

from pipediag.diagnostics.calibration import CalibrationObservation
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.explanation import (
    ExplanationBuilder,
    ExplanationSpec,
)
from pipediag.diagnostics.explanation_calibration import (
    EvidenceInterpretationCalibrator,
)
from pipediag.diagnostics.threshold_reporting import (
    DEFAULT_REPORTING_THRESHOLD,
)


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


def _h3_calibrator():
    observations = [
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=False,
            value=0.05,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=False,
            value=0.08,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=False,
            value=0.10,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=False,
            value=0.12,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=False,
            value=0.15,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=True,
            value=0.55,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=True,
            value=0.65,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=True,
            value=0.75,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=True,
            value=0.85,
            measurement="mean_absolute_standardized_shift",
        ),
        CalibrationObservation(
            evidence_group="G3",
            hypothesis="H3",
            hypothesis_true=True,
            value=0.95,
            measurement="mean_absolute_standardized_shift",
        ),
    ]

    calibrator = EvidenceInterpretationCalibrator()
    calibrator.fit(observations)

    return calibrator


def test_builder_returns_explanation_spec():
    builder = ExplanationBuilder()

    result = builder.build(_result())

    assert isinstance(result, ExplanationSpec)


def test_builder_uses_default_threshold():
    builder = ExplanationBuilder()

    result = builder.build(_result())

    assert result.threshold == DEFAULT_REPORTING_THRESHOLD


def test_builder_reports_hypotheses_above_threshold():
    builder = ExplanationBuilder()

    result = builder.build(_result())

    assert [
        hypothesis.hypothesis_id
        for hypothesis in result.hypotheses
    ] == ["H3"]


def test_reported_hypotheses_are_ranked():
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

    result = builder.build(
        _result(
            posterior=posterior
        )
    )

    assert [
        hypothesis.hypothesis_id
        for hypothesis in result.hypotheses
    ] == [
        "H3",
        "H1",
    ]


def test_uniform_posterior_is_inconclusive():
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

    result = builder.build(
        _result(
            posterior=posterior,
            uncertainty=1.0,
        )
    )

    assert result.is_inconclusive
    assert result.hypotheses == ()
    assert "No dominant diagnostic explanation" in result.overall_assessment


def test_h3_evidence_is_interpreted():
    evidence = EvidenceGroup(
        name="G3",
        value=0.63,
        status="observed",
        metadata={
            "metric": "mean_absolute_standardized_shift"
        },
    )

    builder = ExplanationBuilder()

    result = builder.build(
        _result(
            evidence=[evidence]
        )
    )

    hypothesis = result.hypotheses[0]

    assert hypothesis.hypothesis_id == "H3"
    assert hypothesis.evidence[0].metric == (
        "mean_absolute_standardized_shift"
    )
    assert "substantial" in hypothesis.evidence[0].interpretation


def test_h4_evidence_is_interpreted():
    posterior = {
        "H1": 0.02,
        "H2": 0.02,
        "H3": 0.02,
        "H4": 0.80,
        "H5": 0.02,
        "H6": 0.02,
        "H7": 0.04,
        "H8": 0.03,
        "H9": 0.03,
    }

    evidence = EvidenceGroup(
        name="G4",
        value=0.10,
        status="observed",
        metadata={
            "metric": "minority_class_proportion"
        },
    )

    builder = ExplanationBuilder()

    result = builder.build(
        _result(
            posterior=posterior,
            evidence=[evidence],
        )
    )

    hypothesis = next(
        hypothesis
        for hypothesis in result.hypotheses
        if hypothesis.hypothesis_id == "H4"
    )

    assert "small proportion" in hypothesis.evidence[0].interpretation


def test_h5_zero_missing_values_is_explained():
    posterior = {
        "H1": 0.02,
        "H2": 0.02,
        "H3": 0.02,
        "H4": 0.02,
        "H5": 0.80,
        "H6": 0.02,
        "H7": 0.04,
        "H8": 0.03,
        "H9": 0.03,
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

    result = builder.build(
        _result(
            posterior=posterior,
            evidence=[evidence],
        )
    )

    hypothesis = next(
        hypothesis
        for hypothesis in result.hypotheses
        if hypothesis.hypothesis_id == "H5"
    )

    assert "No missing" in hypothesis.evidence[0].interpretation


def test_h7_uses_training_accuracy():
    posterior = {
        "H1": 0.02,
        "H2": 0.02,
        "H3": 0.02,
        "H4": 0.02,
        "H5": 0.02,
        "H6": 0.02,
        "H7": 0.80,
        "H8": 0.05,
        "H9": 0.03,
    }

    evidence = EvidenceGroup(
        name="G2",
        value=0.05,
        status="observed",
        metadata={
            "measurements": {
                "generalization_gap": 0.05,
                "training_accuracy": 0.60,
                "evaluation_accuracy": 0.55,
            }
        },
    )

    builder = ExplanationBuilder()

    result = builder.build(
        _result(
            posterior=posterior,
            evidence=[evidence],
        )
    )

    hypothesis = next(
        hypothesis
        for hypothesis in result.hypotheses
        if hypothesis.hypothesis_id == "H7"
    )

    metrics = [
        evidence.metric
        for evidence in hypothesis.evidence
    ]

    assert "training_accuracy" in metrics


def test_recommendation_is_present():
    builder = ExplanationBuilder()

    result = builder.build(_result())

    assert result.hypotheses[0].recommendation


def test_explanation_contains_causal_limitation():
    builder = ExplanationBuilder()

    result = builder.build(_result())

    assert any(
        "does not establish a causal root cause" in point
        for point in result.hypotheses[0].explanation_points
    )


def test_uncertainty_is_preserved():
    builder = ExplanationBuilder()

    result = builder.build(
        _result(
            uncertainty=0.73
        )
    )

    assert result.uncertainty == pytest.approx(0.73)


def test_none_result_is_rejected():
    builder = ExplanationBuilder()

    with pytest.raises(ValueError, match="cannot be None"):
        builder.build(None)


def test_unknown_hypothesis_is_rejected():
    posterior = {
        "H10": 1.0,
    }

    builder = ExplanationBuilder()

    with pytest.raises(ValueError, match="Unsupported hypothesis"):
        builder.build(
            _result(
                posterior=posterior
            )
        )


def test_unmapped_evidence_is_not_attached():
    evidence = EvidenceGroup(
        name="UNKNOWN",
        value=0.90,
        status="observed",
        metadata={
            "metric": "unknown_metric"
        },
    )

    builder = ExplanationBuilder()

    result = builder.build(
        _result(
            evidence=[evidence]
        )
    )

    assert result.hypotheses[0].evidence == ()


def test_calibrated_interpretation_overrides_fallback():
    calibrator = _h3_calibrator()

    evidence = EvidenceGroup(
        name="G3",
        value=0.40,
        status="observed",
        metadata={
            "metric": "mean_absolute_standardized_shift"
        },
    )

    builder = ExplanationBuilder(
        interpretation_calibrator=calibrator
    )

    result = builder.build(
        _result(
            evidence=[evidence]
        )
    )

    hypothesis = result.hypotheses[0]

    assert "clearly observable" in hypothesis.evidence[0].interpretation


def test_calibrated_interpreter_is_optional():
    builder = ExplanationBuilder()

    assert builder.interpretation_calibrator is None


def test_calibrated_builder_exposes_calibrator():
    calibrator = _h3_calibrator()

    builder = ExplanationBuilder(
        interpretation_calibrator=calibrator
    )

    assert builder.interpretation_calibrator is calibrator