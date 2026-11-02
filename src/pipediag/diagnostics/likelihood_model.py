from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GaussianLikelihoodModel:
    mean_true: float
    std_true: float
    mean_false: float
    std_false: float
    measurement: str = "default"

    def __post_init__(self):
        if self.std_true <= 0:
            raise ValueError("std_true must be positive")

        if self.std_false <= 0:
            raise ValueError("std_false must be positive")

        if not self.measurement:
            raise ValueError("measurement must not be empty")

        parameters = (
            self.mean_true,
            self.std_true,
            self.mean_false,
            self.std_false,
        )

        if not all(math.isfinite(value) for value in parameters):
            raise ValueError(
                "Likelihood parameters must be finite"
            )

    def _log_likelihood(
        self,
        evidence_value: float,
        hypothesis_true: bool,
    ) -> float:
        if not math.isfinite(evidence_value):
            raise ValueError(
                "Evidence value must be finite"
            )

        if hypothesis_true:
            mean = self.mean_true
            std = self.std_true
        else:
            mean = self.mean_false
            std = self.std_false

        standardized = (
            (evidence_value - mean) / std
        )

        return (
            -math.log(std)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * standardized ** 2
        )

    def likelihood(
        self,
        evidence_value: float,
        hypothesis_true: bool,
    ) -> float:
        log_likelihood = self._log_likelihood(
            evidence_value=evidence_value,
            hypothesis_true=hypothesis_true,
        )

        likelihood = math.exp(log_likelihood)

        if not math.isfinite(likelihood):
            raise ValueError(
                "Likelihood must be finite"
            )

        return likelihood

    def log_likelihood_ratio(
        self,
        evidence_value: float,
    ) -> float:
        """
        Return the log likelihood ratio:

            log(P(E | H=True) / P(E | H=False))

        Computing this directly avoids numerical underflow and
        overflow for very strong evidence.
        """

        log_true = self._log_likelihood(
            evidence_value=evidence_value,
            hypothesis_true=True,
        )

        log_false = self._log_likelihood(
            evidence_value=evidence_value,
            hypothesis_true=False,
        )

        log_ratio = log_true - log_false

        if not math.isfinite(log_ratio):
            raise ValueError(
                "Log likelihood ratio must be finite"
            )

        return log_ratio

    def likelihood_ratio(
        self,
        evidence_value: float,
    ) -> float:
        """
        Return the likelihood ratio:

            P(E | H=True)
            ----------------
            P(E | H=False)

        This method remains available for compatibility.

        Extremely large ratios may not be representable as a
        floating-point value. Call log_likelihood_ratio() when
        the evidence may be extremely strong.
        """

        log_ratio = self.log_likelihood_ratio(
            evidence_value=evidence_value
        )

        try:
            ratio = math.exp(log_ratio)
        except OverflowError as exc:
            raise ValueError(
                "Likelihood ratio is too large to represent; "
                "use log_likelihood_ratio()"
            ) from exc

        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(
                "Likelihood ratio must be finite and positive"
            )

        return ratio