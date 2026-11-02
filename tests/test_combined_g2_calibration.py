from pipediag.diagnostics.likelihood_builder import (
    LikelihoodModelBuilder,
)

from experiments.generalization_calibration import (
    collect_generalization_calibration_observations,
)
from experiments.underfitting_calibration import (
    collect_underfitting_calibration_observations,
)


def _build_registry():
    h2_observations = (
        collect_generalization_calibration_observations(
            n_samples=500,
            n_features=10,
            random_states=(1, 2, 3, 4, 5),
        )
    )

    h7_observations = (
        collect_underfitting_calibration_observations(
            n_samples=500,
            n_features=10,
            random_states=(1, 2, 3, 4, 5),
        )
    )

    return LikelihoodModelBuilder().build(
        h2_observations + h7_observations
    )


def test_combined_registry_contains_h2_model():
    registry = _build_registry()

    assert registry.get("G2", "H2") is not None


def test_combined_registry_contains_h7_model():
    registry = _build_registry()

    assert registry.get("G2", "H7") is not None


def test_h2_model_uses_generalization_gap():
    registry = _build_registry()

    model = registry.get("G2", "H2")

    assert model.measurement == (
        "generalization_gap"
    )


def test_h7_model_uses_training_accuracy():
    registry = _build_registry()

    model = registry.get("G2", "H7")

    assert model.measurement == (
        "training_accuracy"
    )


def test_h2_and_h7_use_different_measurements():
    registry = _build_registry()

    h2_model = registry.get("G2", "H2")
    h7_model = registry.get("G2", "H7")

    assert (
        h2_model.measurement
        != h7_model.measurement
    )