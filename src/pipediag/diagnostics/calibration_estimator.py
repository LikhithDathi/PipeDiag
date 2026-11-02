from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.likelihood_estimator import (
    GaussianLikelihoodEstimator,
    GaussianLikelihoodParameters,
)


class CalibrationEstimator:
    def __init__(
        self,
        likelihood_estimator: GaussianLikelihoodEstimator | None = None,
    ):
        if likelihood_estimator is None:
            likelihood_estimator = GaussianLikelihoodEstimator()

        self.likelihood_estimator = likelihood_estimator

    def fit(
        self,
        observations: list[CalibrationObservation],
    ) -> GaussianLikelihoodParameters:

        if not observations:
            raise ValueError(
                "Calibration observations cannot be empty"
            )

        evidence_groups = {
            observation.evidence_group
            for observation in observations
        }

        hypotheses = {
            observation.hypothesis
            for observation in observations
        }

        if len(evidence_groups) != 1:
            raise ValueError(
                "Calibration observations must belong to "
                "exactly one evidence group"
            )

        if len(hypotheses) != 1:
            raise ValueError(
                "Calibration observations must belong to "
                "exactly one hypothesis"
            )

        true_values = [
            observation.value
            for observation in observations
            if observation.hypothesis_true
        ]

        false_values = [
            observation.value
            for observation in observations
            if not observation.hypothesis_true
        ]

        if not true_values:
            raise ValueError(
                "Calibration requires true-hypothesis observations"
            )

        if not false_values:
            raise ValueError(
                "Calibration requires false-hypothesis observations"
            )

        return self.likelihood_estimator.fit(
            true_values=true_values,
            false_values=false_values,
        )