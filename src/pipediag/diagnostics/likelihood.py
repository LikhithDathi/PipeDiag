from dataclasses import dataclass


@dataclass(frozen=True)
class LikelihoodObservation:
    evidence_group: str
    hypothesis: str
    likelihood: float

    def __post_init__(self):
        if not self.evidence_group:
            raise ValueError(
                "LikelihoodObservation requires a non-empty evidence group"
            )

        if not self.hypothesis:
            raise ValueError(
                "LikelihoodObservation requires a non-empty hypothesis"
            )

        if not 0.0 <= self.likelihood <= 1.0:
            raise ValueError(
                "Likelihood must be between 0 and 1"
            )