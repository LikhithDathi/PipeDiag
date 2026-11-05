
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MultiFaultBayesianInference:
    """
    Multi-fault Bayesian inference with normalized diagnostic support.

    First calculates an independent Bayesian probability for each
    hypothesis with eligible evidence.

    Then normalizes those independent probabilities so that the
    resulting diagnostic support distribution sums to one.

    The independent probabilities are retained separately for
    fault detection.
    """

    hypothesis_ids: tuple[str, ...] | list[str]
    priors: dict[str, float] | None = None

    def __post_init__(self):
        hypothesis_ids = tuple(self.hypothesis_ids)

        if not hypothesis_ids:
            raise ValueError("At least one hypothesis is required")

        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("Hypothesis IDs must be unique")

        object.__setattr__(self, "hypothesis_ids", hypothesis_ids)

        if self.priors is None:
            default_prior = 1.0 / len(hypothesis_ids)

            object.__setattr__(
                self,
                "priors",
                {
                    hypothesis: default_prior
                    for hypothesis in hypothesis_ids
                },
            )
            return

        if set(self.priors) != set(hypothesis_ids):
            raise ValueError(
                "Priors must be provided for exactly "
                "the configured hypotheses"
            )

        validated_priors = {}

        for hypothesis, probability in self.priors.items():
            if not isinstance(probability, (int, float)):
                raise ValueError(
                    "Prior probabilities must be numeric"
                )

            probability = float(probability)

            if (
                not math.isfinite(probability)
                or not 0.0 < probability < 1.0
            ):
                raise ValueError(
                    "Prior probabilities must be finite "
                    "and strictly between 0 and 1"
                )

            validated_priors[hypothesis] = probability

        object.__setattr__(self, "priors", validated_priors)

    def infer_independent_probabilities(
        self,
        evidence_log_ratios: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """
        Calculate independent Bayesian probabilities.

        These probabilities are used for fault detection and are
        not required to sum to one.

        Only hypotheses with eligible evidence are returned.
        """

        log_odds = self._calculate_log_odds(evidence_log_ratios)

        if not log_odds:
            return dict(self.priors)

        return {
            hypothesis: self._sigmoid(value)
            for hypothesis, value in log_odds.items()
        }

    def infer_with_log_likelihood_ratios(
        self,
        evidence_log_ratios: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """
        Calculate normalized multi-fault diagnostic support.

        Stage 1:
            Calculate independent Bayesian posterior probabilities.

        Stage 2:
            Normalize the independent probabilities across
            hypotheses with eligible evidence.

        The returned probabilities sum to one.
        """

        log_odds = self._calculate_log_odds(evidence_log_ratios)

        # With no eligible evidence, return normalized prior weights.
        if not log_odds:
            return self._normalize_weights(self.priors)

        # Calculate log(independent posterior) directly.
        # This avoids underflow when probabilities are extremely small.
        log_scores = {
            hypothesis: self._log_sigmoid(value)
            for hypothesis, value in log_odds.items()
        }

        # Stable softmax normalization over independent probabilities.
        max_score = max(log_scores.values())

        weights = {
            hypothesis: math.exp(score - max_score)
            for hypothesis, score in log_scores.items()
        }

        total = sum(weights.values())

        return {
            hypothesis: weight / total
            for hypothesis, weight in weights.items()
        }

    def _calculate_log_odds(
        self,
        evidence_log_ratios: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate log likelihood ratios for eligible hypotheses."""

        if evidence_log_ratios is None:
            raise ValueError("evidence_log_ratios is required")

        if not isinstance(evidence_log_ratios, dict):
            raise ValueError(
                "evidence_log_ratios must be a dictionary"
            )

        if not evidence_log_ratios:
            return {}

        combined_log_ratios = {
            hypothesis: 0.0
            for hypothesis in self.hypothesis_ids
        }

        eligible_hypotheses = set()

        for evidence_group, ratios in evidence_log_ratios.items():

            if not isinstance(evidence_group, str) or not evidence_group:
                raise ValueError(
                    "Evidence group name must not be empty"
                )

            if ratios is None:
                continue

            if not isinstance(ratios, dict):
                raise ValueError(
                    "Evidence log ratios must be dictionaries"
                )

            if not ratios:
                raise ValueError(
                    "Observed evidence log ratios must not be empty"
                )

            for hypothesis, log_ratio in ratios.items():

                if hypothesis not in combined_log_ratios:
                    raise ValueError(
                        f"Unknown hypothesis '{hypothesis}'"
                    )

                if not isinstance(log_ratio, (int, float)):
                    raise ValueError(
                        "Log likelihood ratios must be numeric"
                    )

                log_ratio = float(log_ratio)

                if not math.isfinite(log_ratio):
                    raise ValueError(
                        "Log likelihood ratios must be finite"
                    )

                combined_log_ratios[hypothesis] += log_ratio
                eligible_hypotheses.add(hypothesis)

        if not eligible_hypotheses:
            return {}

        posterior_log_odds = {}

        for hypothesis in eligible_hypotheses:
            prior = self.priors[hypothesis]

            prior_log_odds = (
                math.log(prior)
                - math.log1p(-prior)
            )

            posterior_log_odds[hypothesis] = (
                prior_log_odds
                + combined_log_ratios[hypothesis]
            )

        return posterior_log_odds

    @staticmethod
    def _normalize_weights(
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Normalize positive weights so their sum is one."""

        total = sum(weights.values())

        if total <= 0.0 or not math.isfinite(total):
            raise ValueError(
                "Normalization requires a positive finite total"
            )

        return {
            key: value / total
            for key, value in weights.items()
        }

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Numerically stable logistic sigmoid."""

        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))

        exponential = math.exp(value)

        return exponential / (1.0 + exponential)

    @staticmethod
    def _log_sigmoid(value: float) -> float:
        """Numerically stable logarithm of the sigmoid."""

        if value >= 0.0:
            return -math.log1p(math.exp(-value))

        return value - math.log1p(math.exp(value))
