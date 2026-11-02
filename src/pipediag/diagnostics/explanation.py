from __future__ import annotations

from dataclasses import dataclass

from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.explanation_calibration import (
    EvidenceInterpretationCalibrator,
)
from pipediag.diagnostics.threshold_reporting import (
    DEFAULT_REPORTING_THRESHOLD,
    ThresholdReporter,
)


@dataclass(frozen=True)
class EvidenceExplanation:
    evidence_group: str
    metric: str
    value: float
    interpretation: str

    def __post_init__(self) -> None:
        if not self.evidence_group:
            raise ValueError(
                "EvidenceExplanation requires a non-empty evidence_group"
            )

        if not self.metric:
            raise ValueError(
                "EvidenceExplanation requires a non-empty metric"
            )

        if not isinstance(self.value, (int, float)):
            raise ValueError(
                "EvidenceExplanation value must be numeric"
            )

        if not self.interpretation:
            raise ValueError(
                "EvidenceExplanation requires a non-empty interpretation"
            )


@dataclass(frozen=True)
class HypothesisExplanation:
    hypothesis_id: str
    title: str
    posterior: float
    evidence: tuple[EvidenceExplanation, ...]
    explanation_points: tuple[str, ...]
    recommendation: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError(
                "HypothesisExplanation requires a non-empty hypothesis_id"
            )

        if not self.title:
            raise ValueError(
                "HypothesisExplanation requires a non-empty title"
            )

        if not 0.0 <= self.posterior <= 1.0:
            raise ValueError(
                "HypothesisExplanation posterior must be between 0 and 1"
            )

        if not self.explanation_points:
            raise ValueError(
                "HypothesisExplanation requires explanation points"
            )

        if not self.recommendation:
            raise ValueError(
                "HypothesisExplanation requires a recommendation"
            )


@dataclass(frozen=True)
class ExplanationSpec:
    threshold: float
    uncertainty: float
    hypotheses: tuple[HypothesisExplanation, ...]
    overall_assessment: str

    @property
    def is_inconclusive(self) -> bool:
        return not self.hypotheses


# ---------------------------------------------------------------------------
# Hypothesis definitions
# ---------------------------------------------------------------------------

HYPOTHESIS_TITLES = {
    "H1": "Data Leakage",
    "H2": "Generalization Failure",
    "H3": "Distribution Shift",
    "H4": "Class Imbalance",
    "H5": "Data Quality Problem",
    "H6": "Evaluation Problem",
    "H7": "Underfitting",
    "H8": "Feature-related Problem",
    "H9": "Pipeline/Execution Problem",
}


HYPOTHESIS_RECOMMENDATIONS = {
    "H1": (
        "Inspect feature construction, preprocessing, and training/evaluation "
        "boundaries for unintended information flow."
    ),
    "H2": (
        "Compare training and evaluation performance and investigate why the "
        "model does not generalize well to unseen data."
    ),
    "H3": (
        "Compare reference and evaluation feature distributions to identify "
        "which features contribute to the observed distribution difference."
    ),
    "H4": (
        "Inspect class proportions and class-specific performance, particularly "
        "for the minority class."
    ),
    "H5": (
        "Inspect the observed data-quality issues, including missing or invalid "
        "values and other integrity measurements."
    ),
    "H6": (
        "Review evaluation sample size, evaluation methodology, and the "
        "reliability of the available evaluation evidence."
    ),
    "H7": (
        "Examine whether the model has sufficient capacity to capture the "
        "predictive structure present in the data."
    ),
    "H8": (
        "Inspect feature representation properties such as constant features, "
        "dimensionality, and redundancy."
    ),
    "H9": (
        "Inspect failed or anomalous execution events and the pipeline "
        "components associated with them."
    ),
}


class ExplanationBuilder:
    """
    Builds a deterministic, evidence-grounded explanation specification.

    This layer does not perform Bayesian inference. It interprets the
    already-computed diagnostic result and produces structured information
    that can later be rendered directly or passed to a constrained LLM.

    If an EvidenceInterpretationCalibrator is supplied, evidence-strength
    labels are derived from empirical calibration observations. Otherwise,
    the built-in deterministic fallback rules are used.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_REPORTING_THRESHOLD,
        interpretation_calibrator: (
            EvidenceInterpretationCalibrator | None
        ) = None,
    ) -> None:
        self.threshold_reporter = ThresholdReporter(
            threshold=threshold
        )

        self.interpretation_calibrator = interpretation_calibrator

    @property
    def threshold(self) -> float:
        return self.threshold_reporter.threshold

    def build(
        self,
        diagnostic_result: DiagnosticResult,
    ) -> ExplanationSpec:
        if diagnostic_result is None:
            raise ValueError(
                "diagnostic_result cannot be None"
            )

        report = self.threshold_reporter.report(
            diagnostic_result
        )

        hypotheses = tuple(
            self._build_hypothesis_explanation(
                hypothesis.hypothesis_id,
                hypothesis.posterior,
                diagnostic_result,
            )
            for hypothesis in report.reported_hypotheses
        )

        if hypotheses:
            overall_assessment = (
                "Potential diagnostic issue(s) identified. "
                "The reported hypotheses reached the posterior "
                "reporting threshold."
            )
        else:
            overall_assessment = (
                "No dominant diagnostic explanation identified. "
                "No hypothesis reached the posterior reporting threshold."
            )

        return ExplanationSpec(
            threshold=self.threshold,
            uncertainty=diagnostic_result.uncertainty,
            hypotheses=hypotheses,
            overall_assessment=overall_assessment,
        )

    def _build_hypothesis_explanation(
        self,
        hypothesis_id: str,
        posterior: float,
        diagnostic_result: DiagnosticResult,
    ) -> HypothesisExplanation:
        if hypothesis_id not in HYPOTHESIS_TITLES:
            raise ValueError(
                f"Unsupported hypothesis: {hypothesis_id}"
            )

        evidence = self._build_evidence_explanations(
            hypothesis_id,
            diagnostic_result,
        )

        explanation_points = self._build_explanation_points(
            hypothesis_id=hypothesis_id,
            posterior=posterior,
            evidence=evidence,
        )

        return HypothesisExplanation(
            hypothesis_id=hypothesis_id,
            title=HYPOTHESIS_TITLES[hypothesis_id],
            posterior=posterior,
            evidence=evidence,
            explanation_points=explanation_points,
            recommendation=HYPOTHESIS_RECOMMENDATIONS[hypothesis_id],
        )

    def _build_evidence_explanations(
        self,
        hypothesis_id: str,
        diagnostic_result: DiagnosticResult,
    ) -> tuple[EvidenceExplanation, ...]:
        explanations: list[EvidenceExplanation] = []

        for evidence_group in diagnostic_result.evidence:
            if evidence_group.name == "G1" and hypothesis_id == "H1":
                explanations.extend(
                    self._explain_g1(evidence_group)
                )

            elif evidence_group.name == "G2" and hypothesis_id in {
                "H2",
                "H7",
            }:
                explanations.extend(
                    self._explain_g2(
                        evidence_group,
                        hypothesis_id,
                    )
                )

            elif evidence_group.name == "G3" and hypothesis_id == "H3":
                explanations.extend(
                    self._explain_g3(evidence_group)
                )

            elif evidence_group.name == "G4" and hypothesis_id == "H4":
                explanations.extend(
                    self._explain_g4(evidence_group)
                )

            elif evidence_group.name == "G5" and hypothesis_id == "H5":
                explanations.extend(
                    self._explain_g5(evidence_group)
                )

            elif evidence_group.name == "G6" and hypothesis_id == "H6":
                explanations.extend(
                    self._explain_g6(evidence_group)
                )

            elif evidence_group.name == "G7" and hypothesis_id == "H8":
                explanations.extend(
                    self._explain_g7(evidence_group)
                )

            elif evidence_group.name == "G8" and hypothesis_id == "H9":
                explanations.extend(
                    self._explain_g8(evidence_group)
                )

        return tuple(explanations)

    # -----------------------------------------------------------------------
    # Calibration-aware interpretation helper
    # -----------------------------------------------------------------------

    def _interpret(
        self,
        *,
        evidence_group: str,
        hypothesis_id: str,
        measurement: str,
        value: float,
        fallback: dict[str, str],
    ) -> str:
        """
        Return an evidence-strength label using calibration when available.

        The returned label is converted into the hypothesis-specific
        explanatory sentence supplied by the caller.

        Calibration is intentionally optional so that ExplanationBuilder
        remains usable without calibration data.
        """
        if self.interpretation_calibrator is not None:
            if self.interpretation_calibrator.contains(
                evidence_group,
                hypothesis_id,
                measurement,
            ):
                return self.interpretation_calibrator.classify(
                    evidence_group=evidence_group,
                    hypothesis=hypothesis_id,
                    measurement=measurement,
                    value=value,
                )

        # Fallback is only used when no calibrated interpretation exists.
        if "substantial" in fallback:
            if fallback["substantial"](value):
                return "substantial"

        if "observable" in fallback:
            if fallback["observable"](value):
                return "observable"

        return "limited"

    # -----------------------------------------------------------------------
    # Evidence-specific interpretation rules
    # -----------------------------------------------------------------------

    def _explain_g1(self, evidence_group) -> list[EvidenceExplanation]:
        metric = self._metric_name(
            evidence_group,
            default="leakage_signal",
        )
        value = float(evidence_group.value)

        strength = self._interpret(
            evidence_group="G1",
            hypothesis_id="H1",
            measurement=metric,
            value=value,
            fallback={
                "substantial": lambda x: x >= 0.75,
                "observable": lambda x: x >= 0.30,
            },
        )

        interpretations = {
            "substantial": (
                "A strong leakage signal was observed in the measured evidence."
            ),
            "observable": (
                "A measurable leakage signal was observed in the measured evidence."
            ),
            "limited": (
                "Only a limited leakage signal was observed in the measured evidence."
            ),
        }

        return [
            EvidenceExplanation(
                evidence_group="G1",
                metric=metric,
                value=value,
                interpretation=interpretations[strength],
            )
        ]

    def _explain_g2(
        self,
        evidence_group,
        hypothesis_id: str,
    ) -> list[EvidenceExplanation]:
        measurements = evidence_group.metadata.get(
            "measurements",
            {},
        )

        explanations: list[EvidenceExplanation] = []

        if "generalization_gap" in measurements:
            gap = self._measurement_value(
                measurements["generalization_gap"]
            )

            strength = self._interpret(
                evidence_group="G2",
                hypothesis_id=hypothesis_id,
                measurement="generalization_gap",
                value=gap,
                fallback={
                    "substantial": lambda x: x >= 0.20,
                    "observable": lambda x: x >= 0.10,
                },
            )

            interpretations = {
                "substantial": (
                    "The training and evaluation performance show a substantial "
                    "generalization gap."
                ),
                "observable": (
                    "The training and evaluation performance show a measurable "
                    "generalization gap."
                ),
                "limited": (
                    "The training and evaluation performance show only a limited "
                    "generalization gap."
                ),
            }

            explanations.append(
                EvidenceExplanation(
                    evidence_group="G2",
                    metric="generalization_gap",
                    value=gap,
                    interpretation=interpretations[strength],
                )
            )

        if "training_accuracy" in measurements:
            training_accuracy = self._measurement_value(
                measurements["training_accuracy"]
            )

            if hypothesis_id == "H7":
                strength = self._interpret(
                    evidence_group="G2",
                    hypothesis_id="H7",
                    measurement="training_accuracy",
                    value=training_accuracy,
                    fallback={
                        "substantial": lambda x: x < 0.70,
                        "observable": lambda x: x < 0.80,
                    },
                )

                interpretations = {
                    "substantial": (
                        "Training accuracy is relatively low, which is consistent "
                        "with insufficient model fit."
                    ),
                    "observable": (
                        "Training accuracy is somewhat reduced and provides "
                        "some evidence of limited model fit."
                    ),
                    "limited": (
                        "Training accuracy is not strongly indicative of poor "
                        "training fit."
                    ),
                }

                explanations.append(
                    EvidenceExplanation(
                        evidence_group="G2",
                        metric="training_accuracy",
                        value=training_accuracy,
                        interpretation=interpretations[strength],
                    )
                )

        if "evaluation_accuracy" in measurements:
            evaluation_accuracy = self._measurement_value(
                measurements["evaluation_accuracy"]
            )

            explanations.append(
                EvidenceExplanation(
                    evidence_group="G2",
                    metric="evaluation_accuracy",
                    value=evaluation_accuracy,
                    interpretation=(
                        "The evaluation accuracy provides additional context "
                        "for interpreting model performance."
                    ),
                )
            )

        return explanations

    def _explain_g3(self, evidence_group) -> list[EvidenceExplanation]:
        value = float(evidence_group.value)
        metric = self._metric_name(
            evidence_group,
            default="mean_absolute_standardized_shift",
        )

        strength = self._interpret(
            evidence_group="G3",
            hypothesis_id="H3",
            measurement=metric,
            value=value,
            fallback={
                "substantial": lambda x: x >= 0.60,
                "observable": lambda x: x >= 0.30,
            },
        )

        interpretations = {
            "substantial": (
                "The measured distribution difference is substantial."
            ),
            "observable": (
                "The measured distribution difference is clearly observable."
            ),
            "limited": (
                "The measured distribution difference is limited."
            ),
        }

        return [
            EvidenceExplanation(
                evidence_group="G3",
                metric=metric,
                value=value,
                interpretation=interpretations[strength],
            )
        ]

    def _explain_g4(self, evidence_group) -> list[EvidenceExplanation]:
        value = float(evidence_group.value)
        metric = self._metric_name(
            evidence_group,
            default="minority_class_proportion",
        )

        strength = self._interpret(
            evidence_group="G4",
            hypothesis_id="H4",
            measurement=metric,
            value=value,
            fallback={
                "substantial": lambda x: x <= 0.15,
                "observable": lambda x: x <= 0.25,
            },
        )

        interpretations = {
            "substantial": (
                "The minority class represents a small proportion of the data."
            ),
            "observable": (
                "The minority class representation is reduced."
            ),
            "limited": (
                "The minority class representation is not strongly reduced "
                "according to the measured proportion."
            ),
        }

        return [
            EvidenceExplanation(
                evidence_group="G4",
                metric=metric,
                value=value,
                interpretation=interpretations[strength],
            )
        ]

    def _explain_g5(self, evidence_group) -> list[EvidenceExplanation]:
        value = float(evidence_group.value)
        metric = self._metric_name(
            evidence_group,
            default="missing_value_rate",
        )

        strength = self._interpret(
            evidence_group="G5",
            hypothesis_id="H5",
            measurement=metric,
            value=value,
            fallback={
                "substantial": lambda x: x >= 0.20,
                "observable": lambda x: x >= 0.05,
            },
        )

        interpretations = {
            "substantial": (
                "A substantial proportion of values were observed to be missing."
            ),
            "observable": (
                "A measurable amount of missing data was observed."
            ),
            "limited": (
                "No missing values were observed in the measured data."
                if value == 0.0
                else "A small amount of missing data was observed."
            ),
        }

        return [
            EvidenceExplanation(
                evidence_group="G5",
                metric=metric,
                value=value,
                interpretation=interpretations[strength],
            )
        ]

    def _explain_g6(self, evidence_group) -> list[EvidenceExplanation]:
        value = float(evidence_group.value)
        metric = self._metric_name(
            evidence_group,
            default="evaluation_standard_error",
        )

        strength = self._interpret(
            evidence_group="G6",
            hypothesis_id="H6",
            measurement=metric,
            value=value,
            fallback={
                "substantial": lambda x: x >= 0.20,
                "observable": lambda x: x >= 0.10,
            },
        )

        interpretations = {
            "substantial": (
                "The evaluation estimate has high statistical uncertainty."
            ),
            "observable": (
                "The evaluation estimate has noticeable statistical uncertainty."
            ),
            "limited": (
                "The evaluation estimate has relatively limited statistical "
                "uncertainty."
            ),
        }

        return [
            EvidenceExplanation(
                evidence_group="G6",
                metric=metric,
                value=value,
                interpretation=interpretations[strength],
            )
        ]

    def _explain_g7(self, evidence_group) -> list[EvidenceExplanation]:
        value = float(evidence_group.value)
        metric = self._metric_name(
            evidence_group,
            default="constant_feature_ratio",
        )

        strength = self._interpret(
            evidence_group="G7",
            hypothesis_id="H8",
            measurement=metric,
            value=value,
            fallback={
                "substantial": lambda x: x >= 0.50,
                "observable": lambda x: x > 0.0,
            },
        )

        interpretations = {
            "substantial": (
                "A substantial proportion of features were measured as constant."
            ),
            "observable": (
                "Some constant features were observed in the feature representation."
            ),
            "limited": (
                "No constant features were observed in the measured feature "
                "representation."
            ),
        }

        return [
            EvidenceExplanation(
                evidence_group="G7",
                metric=metric,
                value=value,
                interpretation=interpretations[strength],
            )
        ]

    def _explain_g8(self, evidence_group) -> list[EvidenceExplanation]:
        value = float(evidence_group.value)
        metric = self._metric_name(
            evidence_group,
            default="execution_failure_rate",
        )

        strength = self._interpret(
            evidence_group="G8",
            hypothesis_id="H9",
            measurement=metric,
            value=value,
            fallback={
                "substantial": lambda x: x >= 0.50,
                "observable": lambda x: x > 0.0,
            },
        )

        interpretations = {
            "substantial": (
                "A high proportion of the observed execution events failed."
            ),
            "observable": (
                "Execution failures were observed during pipeline execution."
            ),
            "limited": (
                "No execution failures were observed in the measured events."
            ),
        }

        return [
            EvidenceExplanation(
                evidence_group="G8",
                metric=metric,
                value=value,
                interpretation=interpretations[strength],
            )
        ]

    # -----------------------------------------------------------------------
    # Hypothesis-level explanation rules
    # -----------------------------------------------------------------------

    def _build_explanation_points(
        self,
        *,
        hypothesis_id: str,
        posterior: float,
        evidence: tuple[EvidenceExplanation, ...],
    ) -> tuple[str, ...]:
        points: list[str] = []

        title = HYPOTHESIS_TITLES[hypothesis_id]

        points.append(
            f"{title} is a potential explanation for the observed "
            f"pipeline behavior, with posterior support of "
            f"{posterior:.2f}."
        )

        if evidence:
            points.append(
                f"The diagnostic result contains {len(evidence)} "
                f"evidence measurement(s) directly associated with "
                f"{title}."
            )
        else:
            points.append(
                "The hypothesis crossed the reporting threshold, but "
                "no directly associated evidence measurement was available "
                "for detailed interpretation."
            )

        points.append(
            "The posterior represents probabilistic diagnostic support "
            "and does not establish a causal root cause."
        )

        return tuple(points)

    @staticmethod
    def _metric_name(
        evidence_group,
        *,
        default: str,
    ) -> str:
        metadata = evidence_group.metadata

        metric = metadata.get("metric")

        if metric:
            return str(metric)

        measurements = metadata.get("measurements", {})

        if len(measurements) == 1:
            return str(next(iter(measurements)))

        return default





    @staticmethod
    def _measurement_value(measurement) -> float:
        """Read a numeric value from either a measurement object or a number."""
        value = (
            measurement.value
            if hasattr(measurement, "value")
            else measurement
        )
        return float(value)