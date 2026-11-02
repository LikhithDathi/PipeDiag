from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.evidence_group import EvidenceGroup


class EvidenceInference:
    def __init__(
        self,
        bayesian_inference: BayesianInference,
    ):
        self.bayesian_inference = bayesian_inference

    def infer(
        self,
        evidence_groups: list[EvidenceGroup],
        likelihoods: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        if likelihoods is None:
            likelihoods = {}

        evidence_names = [
            evidence.name
            for evidence in evidence_groups
        ]

        if len(evidence_names) != len(set(evidence_names)):
            raise ValueError(
                "Evidence group names must be unique"
            )

        evidence_by_name = {
            evidence.name: evidence
            for evidence in evidence_groups
        }

        for evidence in evidence_groups:
            if evidence.status == "unavailable":
                continue

            if evidence.status != "observed":
                raise ValueError(
                    "Evidence status must be either "
                    "'observed' or 'unavailable'"
                )

            if evidence.name not in likelihoods:
                raise ValueError(
                    "Observed evidence is missing a likelihood model: "
                    f"{evidence.name}"
                )

        unknown_likelihoods = (
            set(likelihoods) - set(evidence_by_name)
        )

        if unknown_likelihoods:
            raise ValueError(
                "Likelihoods were provided for unknown evidence groups: "
                f"{sorted(unknown_likelihoods)}"
            )

        observed_likelihoods = {}

        for evidence in evidence_groups:
            if evidence.status != "observed":
                continue

            hypothesis_likelihoods = likelihoods[evidence.name]

            observed_likelihoods[evidence.name] = (
                self._validate_hypothesis_likelihoods(
                    hypothesis_likelihoods
                )
            )

        if not observed_likelihoods:
            return dict(self.bayesian_inference.priors)

        combined_likelihoods = {}

        for hypothesis_id in self.bayesian_inference.hypothesis_ids:
            combined_likelihood = 1.0

            for hypothesis_likelihoods in (
                observed_likelihoods.values()
            ):
                combined_likelihood *= hypothesis_likelihoods[
                    hypothesis_id
                ]

            combined_likelihoods[hypothesis_id] = (
                combined_likelihood
            )

        return self.bayesian_inference.infer(
            combined_likelihoods
        )

    def _validate_hypothesis_likelihoods(
        self,
        likelihoods: dict[str, float],
    ) -> dict[str, float]:
        hypothesis_ids = set(
            self.bayesian_inference.hypothesis_ids
        )

        provided_ids = set(likelihoods)

        if provided_ids != hypothesis_ids:
            missing = hypothesis_ids - provided_ids
            unknown = provided_ids - hypothesis_ids

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
                "Each observed evidence group must provide "
                "exactly one likelihood for every hypothesis; "
                + "; ".join(messages)
            )

        return dict(likelihoods)