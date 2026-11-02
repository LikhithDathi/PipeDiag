
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MultivariateGaussianLikelihoodModel:
    mean_true: tuple[float, ...]
    covariance_true: tuple[tuple[float, ...], ...]
    mean_false: tuple[float, ...]
    covariance_false: tuple[tuple[float, ...], ...]
    measurements: tuple[str, ...]
    regularization: float = 1e-4

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("measurements must not be empty")

        if len(set(self.measurements)) != len(self.measurements):
            raise ValueError("measurements must be unique")

        if any(not name for name in self.measurements):
            raise ValueError("measurement names must not be empty")

        if self.regularization <= 0 or not math.isfinite(self.regularization):
            raise ValueError("regularization must be finite and positive")

        dimension = len(self.measurements)

        if len(self.mean_true) != dimension:
            raise ValueError("mean_true dimension does not match measurements")

        if len(self.mean_false) != dimension:
            raise ValueError("mean_false dimension does not match measurements")

        for name, mean, covariance in (
            ("true", self.mean_true, self.covariance_true),
            ("false", self.mean_false, self.covariance_false),
        ):
            if len(covariance) != dimension or any(
                len(row) != dimension for row in covariance
            ):
                raise ValueError(f"{name} covariance has incorrect dimensions")

            if not all(math.isfinite(value) for value in mean):
                raise ValueError(f"{name} mean must contain finite values")

            matrix = np.asarray(covariance, dtype=float)

            if not np.isfinite(matrix).all():
                raise ValueError(f"{name} covariance must be finite")

            if not np.allclose(matrix, matrix.T, atol=1e-10):
                raise ValueError(f"{name} covariance must be symmetric")

            regularized = matrix + np.eye(dimension) * self.regularization

            try:
                np.linalg.cholesky(regularized)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"{name} covariance must be positive semidefinite "
                    "after regularization"
                ) from exc

    def _log_density(
        self,
        values: tuple[float, ...],
        mean_values: tuple[float, ...],
        covariance_values: tuple[tuple[float, ...], ...],
    ) -> float:
        if len(values) != len(self.measurements):
            raise ValueError("evidence vector has incorrect dimensions")

        x = np.asarray(values, dtype=float)
        mean = np.asarray(mean_values, dtype=float)
        covariance = np.asarray(covariance_values, dtype=float)

        if not np.isfinite(x).all():
            raise ValueError("evidence values must be finite")

        covariance = (
            covariance
            + np.eye(len(self.measurements)) * self.regularization
        )

        sign, logdet = np.linalg.slogdet(covariance)

        if sign <= 0 or not math.isfinite(float(logdet)):
            raise ValueError("regularized covariance must be positive definite")

        delta = x - mean

        try:
            quadratic = float(delta @ np.linalg.solve(covariance, delta))
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance solve failed") from exc

        result = -0.5 * (
            len(self.measurements) * math.log(2.0 * math.pi)
            + float(logdet)
            + quadratic
        )

        if not math.isfinite(result):
            raise ValueError("log likelihood must be finite")

        return result

    def log_likelihood_ratio(
        self,
        values: tuple[float, ...],
    ) -> float:
        log_true = self._log_density(
            values,
            self.mean_true,
            self.covariance_true,
        )
        log_false = self._log_density(
            values,
            self.mean_false,
            self.covariance_false,
        )

        result = log_true - log_false

        if not math.isfinite(result):
            raise ValueError("log likelihood ratio must be finite")

        return result

    def likelihood_ratio(
        self,
        values: tuple[float, ...],
    ) -> float:
        log_ratio = self.log_likelihood_ratio(values)

        try:
            ratio = math.exp(log_ratio)
        except OverflowError as exc:
            raise ValueError(
                "likelihood ratio is too large; use log_likelihood_ratio"
            ) from exc

        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(
                "likelihood ratio is not representable; "
                "use log_likelihood_ratio"
            )

        return ratio







    def log_likelihood(
        self,
        values: tuple[float, ...],
        hypothesis_true: bool,
    ) -> float:
        if hypothesis_true:
            return self._log_density(
                values,
                self.mean_true,
                self.covariance_true,
            )

        return self._log_density(
            values,
            self.mean_false,
            self.covariance_false,
        )


    def likelihood(
        self,
        values: tuple[float, ...],
        hypothesis_true: bool,
    ) -> float:
        result = math.exp(
            self.log_likelihood(values, hypothesis_true)
        )

        if not math.isfinite(result):
            raise ValueError("Likelihood must be finite")

        return result