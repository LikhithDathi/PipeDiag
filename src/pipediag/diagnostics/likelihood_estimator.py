from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GaussianLikelihoodParameters:
    mean_true: float
    std_true: float
    mean_false: float
    std_false: float


class GaussianLikelihoodEstimator:
    def fit(
        self,
        true_values: list[float],
        false_values: list[float],
    ) -> GaussianLikelihoodParameters:
        self._validate_values(
            true_values,
            "true_values",
        )

        self._validate_values(
            false_values,
            "false_values",
        )

        if len(true_values) < 2:
            raise ValueError(
                "At least two true samples are required"
            )

        if len(false_values) < 2:
            raise ValueError(
                "At least two false samples are required"
            )

        mean_true = self._mean(true_values)
        mean_false = self._mean(false_values)

        std_true = self._population_std(
            true_values,
            mean_true,
        )

        std_false = self._population_std(
            false_values,
            mean_false,
        )

        if math.isclose(std_true, 0.0, abs_tol=1e-12):
            raise ValueError(
                "True values must have non-zero variance"
            )

        if math.isclose(std_false, 0.0, abs_tol=1e-12):
            raise ValueError(
                "False values must have non-zero variance"
            )

        return GaussianLikelihoodParameters(
            mean_true=mean_true,
            std_true=std_true,
            mean_false=mean_false,
            std_false=std_false,
        )

    @staticmethod
    def _validate_values(
        values: list[float],
        name: str,
    ) -> None:
        if not values:
            raise ValueError(
                f"{name} cannot be empty"
            )

        for value in values:
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must contain only finite values"
                )

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _population_std(
        values: list[float],
        mean: float,
    ) -> float:
        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        return math.sqrt(variance)