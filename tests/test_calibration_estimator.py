import pytest

from pipediag.diagnostics.calibration import (
    CalibrationObservation,
)
from pipediag.diagnostics.calibration_estimator import (
    CalibrationEstimator,
)
from pipediag.diagnostics.likelihood_estimator import (
    GaussianLikelihoodParameters,
)


def create_observations():
    return [
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.8,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.9,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.7,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.2,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.3,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.1,
        ),
    ]


def test_estimator_returns_parameters():
    estimator = CalibrationEstimator()

    result = estimator.fit(
        create_observations()
    )

    assert isinstance(
        result,
        GaussianLikelihoodParameters,
    )


def test_estimator_computes_true_mean():
    estimator = CalibrationEstimator()

    result = estimator.fit(
        create_observations()
    )

    assert result.mean_true == pytest.approx(
        0.8
    )


def test_estimator_computes_false_mean():
    estimator = CalibrationEstimator()

    result = estimator.fit(
        create_observations()
    )

    assert result.mean_false == pytest.approx(
        0.2
    )


def test_estimator_computes_true_standard_deviation():
    estimator = CalibrationEstimator()

    result = estimator.fit(
        create_observations()
    )

    assert result.std_true == pytest.approx(
        ((
            (0.8 - 0.8) ** 2
            + (0.9 - 0.8) ** 2
            + (0.7 - 0.8) ** 2
        ) / 3) ** 0.5
    )


def test_estimator_computes_false_standard_deviation():
    estimator = CalibrationEstimator()

    result = estimator.fit(
        create_observations()
    )

    assert result.std_false == pytest.approx(
        ((
            (0.2 - 0.2) ** 2
            + (0.3 - 0.2) ** 2
            + (0.1 - 0.2) ** 2
        ) / 3) ** 0.5
    )


def test_estimator_rejects_empty_observations():
    estimator = CalibrationEstimator()

    with pytest.raises(ValueError):
        estimator.fit([])


def test_estimator_rejects_only_true_observations():
    estimator = CalibrationEstimator()

    observations = [
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.8,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.9,
        ),
    ]

    with pytest.raises(ValueError):
        estimator.fit(observations)


def test_estimator_rejects_only_false_observations():
    estimator = CalibrationEstimator()

    observations = [
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.2,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.3,
        ),
    ]

    with pytest.raises(ValueError):
        estimator.fit(observations)


def test_estimator_rejects_multiple_evidence_groups():
    estimator = CalibrationEstimator()

    observations = create_observations()

    observations.append(
        CalibrationObservation(
            evidence_group="G2",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.6,
        )
    )

    with pytest.raises(ValueError):
        estimator.fit(observations)


def test_estimator_rejects_multiple_hypotheses():
    estimator = CalibrationEstimator()

    observations = create_observations()

    observations.append(
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H2",
            hypothesis_true=True,
            value=0.6,
        )
    )

    with pytest.raises(ValueError):
        estimator.fit(observations)


def test_estimator_rejects_zero_true_variance():
    estimator = CalibrationEstimator()

    observations = [
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.8,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.8,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.2,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.3,
        ),
    ]

    with pytest.raises(ValueError):
        estimator.fit(observations)


def test_estimator_rejects_zero_false_variance():
    estimator = CalibrationEstimator()

    observations = [
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.8,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=True,
            value=0.9,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.2,
        ),
        CalibrationObservation(
            evidence_group="G1",
            hypothesis="H1",
            hypothesis_true=False,
            value=0.2,
        ),
    ]

    with pytest.raises(ValueError):
        estimator.fit(observations)