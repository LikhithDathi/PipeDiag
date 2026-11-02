import math


def posterior_uncertainty(
    posterior: dict[str, float],
) -> float:
    if not posterior:
        raise ValueError(
            "Posterior distribution cannot be empty"
        )

    probabilities = list(posterior.values())

    for probability in probabilities:
        if not math.isfinite(probability):
            raise ValueError(
                "Posterior probabilities must be finite"
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Posterior probabilities must be between 0 and 1"
            )

    if not math.isclose(
        sum(probabilities),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Posterior probabilities must sum to one"
        )

    entropy = 0.0

    for probability in probabilities:
        if probability > 0.0:
            entropy -= probability * math.log(probability)

    maximum_entropy = math.log(len(probabilities))

    if maximum_entropy == 0.0:
        return 0.0

    return entropy / maximum_entropy