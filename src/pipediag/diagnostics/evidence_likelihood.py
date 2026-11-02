from __future__ import annotations

import math
from numbers import Real

from .evidence_group import EvidenceGroup
from .likelihood_registry import LikelihoodModelRegistry


from .multivariate_likelihood_model import (
    MultivariateGaussianLikelihoodModel,
)


class EvidenceLikelihoodEvaluator:
    def __init__(
        self,
        registry,
        evidence_hypothesis_map,
    ):
        if registry is None:
            raise ValueError("registry is required")

        if evidence_hypothesis_map is None:
            raise ValueError(
                "evidence_hypothesis_map is required"
            )

        self.registry = registry
        self.evidence_hypothesis_map = evidence_hypothesis_map

    def evaluate(
        self,
        evidence: EvidenceGroup,
    ) -> dict[str, float]:
        if evidence.status == "unavailable":
            return {}

        self._validate_evidence(evidence)

        hypotheses = self._mapped_hypotheses(
            evidence.name
        )

        likelihoods: dict[str, float] = {}

        for hypothesis in hypotheses:
            model = self.registry.get(
                evidence.name,
                hypothesis,
            )

            if model is None:
                raise ValueError(
                    f"No likelihood model registered "
                    f"for evidence group '{evidence.name}' "
                    f"and hypothesis '{hypothesis}'"
                )

            value = self._get_model_values(evidence, model)

            likelihood = model.likelihood(
                value,
                hypothesis_true=True,
            )

            if not math.isfinite(likelihood) or likelihood < 0:
                raise ValueError(
                    "Likelihood must be finite and non-negative"
                )

            likelihoods[hypothesis] = likelihood

        return likelihoods

    def evaluate_ratios(
        self,
        evidence: EvidenceGroup,
    ) -> dict[str, float]:
        if evidence.status == "unavailable":
            return {}

        self._validate_evidence(evidence)

        hypotheses = self._mapped_hypotheses(
            evidence.name
        )

        ratios: dict[str, float] = {}

        for hypothesis in hypotheses:
            model = self.registry.get(
                evidence.name,
                hypothesis,
            )

            if model is None:
                raise ValueError(
                    f"No likelihood model registered "
                    f"for evidence group '{evidence.name}' "
                    f"and hypothesis '{hypothesis}'"
                )

            value = self._get_model_values(evidence, model)

            ratio = model.likelihood_ratio(value)

            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(
                    "Likelihood ratio must be finite and positive"
                )

            ratios[hypothesis] = ratio

        return ratios

    def evaluate_log_ratios(
        self,
        evidence: EvidenceGroup,
    ) -> dict[str, float]:
        """
        Evaluate evidence as log likelihood ratios.

        This is the numerically stable path used by production
        Bayesian inference.
        """

        if evidence.status == "unavailable":
            return {}

        self._validate_evidence(evidence)

        hypotheses = self._mapped_hypotheses(
            evidence.name
        )

        log_ratios: dict[str, float] = {}

        for hypothesis in hypotheses:
            model = self.registry.get(
                evidence.name,
                hypothesis,
            )

            if model is None:
                raise ValueError(
                    f"No likelihood model registered "
                    f"for evidence group '{evidence.name}' "
                    f"and hypothesis '{hypothesis}'"
                )

            value = self._get_model_values(evidence, model)

            log_ratio = model.log_likelihood_ratio(value)

            if not math.isfinite(log_ratio):
                raise ValueError(
                    "Log likelihood ratio must be finite"
                )

            log_ratios[hypothesis] = log_ratio

        return log_ratios

    def _get_measurement_value(
        self,
        evidence: EvidenceGroup,
        measurement: str,
    ) -> float:
        if measurement == "default":
            value = evidence.value

        else:
            measurements = evidence.metadata.get(
                "measurements"
            )

            if measurements is None:
                raise ValueError(
                    f"Evidence group '{evidence.name}' "
                    f"does not provide measurements"
                )

            if measurement not in measurements:
                raise ValueError(
                    f"Measurement '{measurement}' is not "
                    f"available for evidence group "
                    f"'{evidence.name}'"
                )

            measurement_object = measurements[measurement]

            if hasattr(measurement_object, "value"):
                value = measurement_object.value
            else:
                value = measurement_object

        if not isinstance(value, Real):
            raise ValueError(
                "Evidence measurement value must be numeric"
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                "Evidence measurement value must be finite"
            )

        return value

    def _validate_evidence(
        self,
        evidence: EvidenceGroup,
    ) -> None:
        if evidence.status != "observed":
            raise ValueError(
                "Evidence status must be "
                "'observed' or 'unavailable'"
            )

        if evidence.name not in self.evidence_hypothesis_map:
            raise ValueError(
                f"Evidence group '{evidence.name}' "
                f"is not present in the evidence-hypothesis map"
            )

    def _mapped_hypotheses(
        self,
        evidence_group: str,
    ) -> set[str]:
        mapping = self.evidence_hypothesis_map[
            evidence_group
        ]

        primary = mapping.get(
            "primary",
            set(),
        )

        secondary = mapping.get(
            "secondary",
            set(),
        )

        return set(primary) | set(secondary)






    def _get_model_values(
        self,
        evidence: EvidenceGroup,
        model,
    ) -> float | tuple[float, ...]:
        if isinstance(model, MultivariateGaussianLikelihoodModel):
            return tuple(
                self._get_measurement_value(evidence, name)
                for name in model.measurements
            )

        return self._get_measurement_value(
            evidence,
            model.measurement,
        )