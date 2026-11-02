from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)


from pipediag.diagnostics.multivariate_likelihood_model import (
    MultivariateGaussianLikelihoodModel,
)


LikelihoodModel = (
    GaussianLikelihoodModel
    | MultivariateGaussianLikelihoodModel
)

class LikelihoodModelRegistry:
    def __init__(self):
        self._models: dict[
            tuple[str, str],
            LikelihoodModel,
        ] = {}

    def register(
        self,
        evidence_group: str,
        hypothesis: str,
        model: LikelihoodModel,
    ) -> None:
        if not evidence_group:
            raise ValueError(
                "Evidence group must be non-empty"
            )

        if not hypothesis:
            raise ValueError(
                "Hypothesis must be non-empty"
            )

        if model is None:
            raise ValueError(
                "Likelihood model cannot be None"
            )

        key = (
            evidence_group,
            hypothesis,
        )

        if key in self._models:
            raise ValueError(
                "A likelihood model is already registered for "
                f"{evidence_group} and {hypothesis}"
            )

        self._models[key] = model

    def get(
        self,
        evidence_group: str,
        hypothesis: str,
    ) -> LikelihoodModel | None:
        return self._models.get(
            (
                evidence_group,
                hypothesis,
            )
        )

    def contains(
        self,
        evidence_group: str,
        hypothesis: str,
    ) -> bool:
        return (
            evidence_group,
            hypothesis,
        ) in self._models

    def models(
        self,
    ) -> dict[
        tuple[str, str],
        LikelihoodModel,
    ]:
        return dict(self._models)