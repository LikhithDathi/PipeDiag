from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BayesianInference:
    hypothesis_ids: list[str]
    priors: dict[str, float] | None = None

    def __post_init__(self):
        if not self.hypothesis_ids:
            raise ValueError(
                "At least one hypothesis is required"
            )

        if len(self.hypothesis_ids) != len(
            set(self.hypothesis_ids)
        ):
            raise ValueError(
                "Hypothesis IDs must be unique"
            )

        if self.priors is None:
            probability = 1.0 / len(self.hypothesis_ids)

            object.__setattr__(
                self,
                "priors",
                {
                    hypothesis_id: probability
                    for hypothesis_id in self.hypothesis_ids
                },
            )
            return

        if set(self.priors) != set(self.hypothesis_ids):
            raise ValueError(
                "Priors must be provided for exactly "
                "the configured hypotheses"
            )

        if any(
            not math.isfinite(probability)
            for probability in self.priors.values()
        ):
            raise ValueError(
                "Prior probabilities must be finite"
            )

        if any(
            probability < 0.0
            for probability in self.priors.values()
        ):
            raise ValueError(
                "Prior probabilities must be non-negative"
            )

        if not math.isclose(
            sum(self.priors.values()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Prior probabilities must sum to one"
            )

    def infer(
        self,
        likelihoods: dict[str, float],
    ) -> dict[str, float]:
        if set(likelihoods) != set(self.hypothesis_ids):
            missing = (
                set(self.hypothesis_ids)
                - set(likelihoods)
            )

            unknown = (
                set(likelihoods)
                - set(self.hypothesis_ids)
            )

            messages = []

            if missing:
                messages.append(
                    f"missing hypotheses: {sorted(missing)}"
                )

            if unknown:
                messages.append(
                    f"unknown hypotheses: {sorted(unknown)}"
                )

            raise ValueError(
                "Likelihoods must contain exactly one value "
                "for every configured hypothesis; "
                + "; ".join(messages)
            )

        for likelihood in likelihoods.values():
            if not math.isfinite(likelihood):
                raise ValueError(
                    "Likelihoods must be finite"
                )

            if not 0.0 <= likelihood <= 1.0:
                raise ValueError(
                    "Likelihoods must be between 0 and 1"
                )

        unnormalized = {
            hypothesis_id: (
                self.priors[hypothesis_id]
                * likelihoods[hypothesis_id]
            )
            for hypothesis_id in self.hypothesis_ids
        }

        total = sum(unnormalized.values())

        if total <= 0.0:
            raise ValueError(
                "Likelihood evidence produces zero total probability"
            )

        return {
            hypothesis_id: value / total
            for hypothesis_id, value in unnormalized.items()
        }