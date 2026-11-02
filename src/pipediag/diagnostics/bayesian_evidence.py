from __future__ import annotations

import math


class BayesianEvidenceInference:

    def __init__(self, bayesian_inference):
        if bayesian_inference is None:
            raise ValueError("bayesian_inference is required")

        self.bayesian_inference = bayesian_inference

    def infer(self, evidence_likelihoods):
        if not evidence_likelihoods:
            return dict(self.bayesian_inference.priors)

        hypothesis_ids = set(
            self.bayesian_inference.hypothesis_ids
        )

        combined_likelihoods = {
            hypothesis: 1.0
            for hypothesis in hypothesis_ids
        }

        observed_evidence = False

        for evidence_group, likelihoods in evidence_likelihoods.items():

            if not evidence_group:
                raise ValueError(
                    "Evidence group name must not be empty"
                )

            if likelihoods is None:
                continue

            if not likelihoods:
                raise ValueError(
                    "Observed evidence likelihoods "
                    "must not be empty"
                )

            observed_evidence = True

            self._validate_likelihoods(
                likelihoods,
                hypothesis_ids,
            )

            for hypothesis, likelihood in likelihoods.items():
                combined_likelihoods[hypothesis] *= likelihood

        if not observed_evidence:
            return dict(self.bayesian_inference.priors)

        unnormalized = {
            hypothesis:
                self.bayesian_inference.priors[hypothesis]
                * combined_likelihoods[hypothesis]
            for hypothesis in hypothesis_ids
        }

        total = sum(unnormalized.values())

        if not math.isfinite(total) or total <= 0:
            raise ValueError(
                "Invalid Bayesian normalization constant"
            )

        return {
            hypothesis: value / total
            for hypothesis, value in unnormalized.items()
        }

    def infer_with_likelihood_ratios(
        self,
        evidence_likelihood_ratios,
    ):
        """
        Backwards-compatible likelihood-ratio inference.

        Converts likelihood ratios to log likelihood ratios and delegates
        to the numerically stable log-space implementation.
        """

        if not evidence_likelihood_ratios:
            return dict(self.bayesian_inference.priors)

        converted = {}

        for evidence_group, ratios in evidence_likelihood_ratios.items():
            if not evidence_group:
                raise ValueError(
                    "Evidence group name must not be empty"
                )

            if ratios is None:
                continue

            if not ratios:
                raise ValueError(
                    "Observed evidence likelihood ratios "
                    "must not be empty"
                )

            log_ratios = {}

            for hypothesis, ratio in ratios.items():
                if hypothesis not in self.bayesian_inference.hypothesis_ids:
                    raise ValueError(
                        f"Unknown hypothesis '{hypothesis}'"
                    )

                if not isinstance(ratio, (int, float)):
                    raise ValueError(
                        "Likelihood ratios must be numeric"
                    )

                ratio = float(ratio)

                if not math.isfinite(ratio):
                    raise ValueError(
                        "Likelihood ratios must be finite"
                    )

                if ratio <= 0:
                    raise ValueError(
                        "Likelihood ratios must be positive"
                    )

                log_ratio = math.log(ratio)

                if not math.isfinite(log_ratio):
                    raise ValueError(
                        "Log likelihood ratio must be finite"
                    )

                log_ratios[hypothesis] = log_ratio

            if log_ratios:
                converted[evidence_group] = log_ratios

        return self.infer_with_log_likelihood_ratios(converted)

    def infer_with_log_likelihood_ratios(
        self,
        evidence_log_ratios,
    ):
        """
        Infer posterior probabilities directly in log space.

        This avoids overflow/underflow when likelihood ratios
        become extremely large or small.
        """

        if not evidence_log_ratios:
            return dict(self.bayesian_inference.priors)

        hypothesis_ids = set(
            self.bayesian_inference.hypothesis_ids
        )

        combined_log_ratios = {
            hypothesis: 0.0
            for hypothesis in hypothesis_ids
        }

        observed_evidence = False

        for evidence_group, log_ratios in (
            evidence_log_ratios.items()
        ):

            if not evidence_group:
                raise ValueError(
                    "Evidence group name must not be empty"
                )

            if log_ratios is None:
                continue

            if not log_ratios:
                raise ValueError(
                    "Observed evidence log ratios "
                    "must not be empty"
                )

            observed_evidence = True

            for hypothesis, log_ratio in log_ratios.items():

                if hypothesis not in hypothesis_ids:
                    raise ValueError(
                        f"Unknown hypothesis '{hypothesis}'"
                    )

                if not isinstance(log_ratio, (int, float)):
                    raise ValueError(
                        "Log likelihood ratios "
                        "must be numeric"
                    )

                log_ratio = float(log_ratio)

                if not math.isfinite(log_ratio):
                    raise ValueError(
                        "Log likelihood ratios "
                        "must be finite"
                    )

                combined_log_ratios[hypothesis] += log_ratio

        if not observed_evidence:
            return dict(self.bayesian_inference.priors)

        log_scores = {}

        for hypothesis in hypothesis_ids:

            prior = self.bayesian_inference.priors[
                hypothesis
            ]

            if not isinstance(prior, (int, float)):
                raise ValueError("Priors must be numeric")

            prior = float(prior)

            if not math.isfinite(prior) or prior <= 0:
                raise ValueError(
                    "Priors must be finite and positive"
                )

            log_scores[hypothesis] = (
                math.log(prior)
                + combined_log_ratios[hypothesis]
            )

        # Stable softmax:
        # subtract the largest log-score before exponentiating.
        maximum = max(log_scores.values())

        exponentials = {
            hypothesis: math.exp(score - maximum)
            for hypothesis, score in log_scores.items()
        }

        total = sum(exponentials.values())

        if not math.isfinite(total) or total <= 0:
            raise ValueError(
                "Invalid Bayesian normalization constant"
            )

        return {
            hypothesis: value / total
            for hypothesis, value in exponentials.items()
        }

    @staticmethod
    def _validate_likelihoods(
        likelihoods,
        hypothesis_ids,
    ):
        for hypothesis, likelihood in likelihoods.items():

            if hypothesis not in hypothesis_ids:
                raise ValueError(
                    f"Unknown hypothesis '{hypothesis}'"
                )

            if not isinstance(likelihood, (int, float)):
                raise ValueError(
                    "Likelihoods must be numeric"
                )

            likelihood = float(likelihood)

            if not math.isfinite(likelihood):
                raise ValueError(
                    "Likelihoods must be finite"
                )

            if likelihood < 0:
                raise ValueError(
                    "Likelihoods must be non-negative"
                )