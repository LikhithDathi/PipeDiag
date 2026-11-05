from __future__ import annotations

from collections import defaultdict

from .calibration import CalibrationObservation
from .likelihood_estimator import (
    GaussianLikelihoodEstimator,
)
from .likelihood_model import (
    GaussianLikelihoodModel,
)
from .likelihood_registry import (
    LikelihoodModelRegistry,
)


class LikelihoodModelBuilder:

    def __init__(
        self,
        estimator: GaussianLikelihoodEstimator | None = None,
    ):
        if estimator is None:
            estimator = GaussianLikelihoodEstimator()

        self.estimator = estimator

    def build(
        self,
        observations: list[CalibrationObservation],
    ) -> LikelihoodModelRegistry:

        if not observations:
            raise ValueError(
                "observations must not be empty"
            )

        grouped: dict[
            tuple[str, str, str],
            list[CalibrationObservation],
        ] = defaultdict(list)

        for observation in observations:
            grouped[
                (
                    observation.evidence_group,
                    observation.hypothesis,
                    observation.measurement,
                )
            ].append(observation)

        registry = LikelihoodModelRegistry()

        for (
            evidence_group,
            hypothesis,
            measurement,
        ), group_observations in grouped.items():

            true_values = [
                observation.value
                for observation in group_observations
                if observation.hypothesis_true
            ]

            false_values = [
                observation.value
                for observation in group_observations
                if not observation.hypothesis_true
            ]

            estimator = self.estimator

            if measurement == "runtime_contamination_rate":
                estimator = GaussianLikelihoodEstimator(
                    min_std=0.005
                )

            parameters = estimator.fit(
                true_values=true_values,
                false_values=false_values,
            )

            model = GaussianLikelihoodModel(
                mean_true=parameters.mean_true,
                std_true=parameters.std_true,
                mean_false=parameters.mean_false,
                std_false=parameters.std_false,
                measurement=measurement,
            )

            registry.register(
                evidence_group=evidence_group,
                hypothesis=hypothesis,
                model=model,
            )

        return registry