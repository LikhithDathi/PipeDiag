from __future__ import annotations

from .likelihood_model import GaussianLikelihoodModel
from .multivariate_likelihood_model import (
    MultivariateGaussianLikelihoodModel,
)

LikelihoodModel = (
    GaussianLikelihoodModel
    | MultivariateGaussianLikelihoodModel
)


class LikelihoodModelRegistry:

    def __init__(self, preserve_measurement_keys: bool = False):
        self._models: dict[
            tuple[str, str, str],
            LikelihoodModel,
        ] = {}
        self.preserve_measurement_keys = preserve_measurement_keys

    def register(
        self,
        evidence_group: str,
        hypothesis: str,
        model: LikelihoodModel,
    ) -> None:
        if not evidence_group or not hypothesis:
            raise ValueError(
                "evidence_group and hypothesis must not be empty"
            )

        if model is None:
            raise ValueError("model must not be None")

        if not isinstance(
            model,
            (
                GaussianLikelihoodModel,
                MultivariateGaussianLikelihoodModel,
            ),
        ):
            raise TypeError("Unsupported likelihood model")

        measurement = (
            model.measurement
            if isinstance(model, GaussianLikelihoodModel)
            else "multivariate"
        )

        key = (evidence_group, hypothesis, measurement)

        if key in self._models:
            raise ValueError(
                f"A likelihood model is already registered for {key}"
            )

        self._models[key] = model

    def get(
        self,
        evidence_group: str,
        hypothesis: str,
        measurement: str | None = None,
    ) -> LikelihoodModel | None:

        if measurement is not None:
            return self._models.get(
                (evidence_group, hypothesis, measurement)
            )

        default = self._models.get(
            (evidence_group, hypothesis, "default")
        )

        if default is not None:
            return default

        matches = [
            model
            for (group, target, _), model in self._models.items()
            if group == evidence_group and target == hypothesis
        ]

        return matches[0] if len(matches) == 1 else None

    def contains(
        self,
        evidence_group: str,
        hypothesis: str,
        measurement: str | None = None,
    ) -> bool:
        return self.get(
            evidence_group,
            hypothesis,
            measurement,
        ) is not None

    def models(self):
        if self.preserve_measurement_keys:
            return dict(self._models)

        counts = {}
        for group, hypothesis, _ in self._models:
            pair = (group, hypothesis)
            counts[pair] = counts.get(pair, 0) + 1

        result = {}

        for (group, hypothesis, measurement), model in (
            self._models.items()
        ):
            pair = (group, hypothesis)

            if counts[pair] == 1:
                key = pair
            else:
                key = (group, hypothesis, measurement)

            result[key] = model

        return result