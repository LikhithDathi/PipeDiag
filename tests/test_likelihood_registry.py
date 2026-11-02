import pytest

from pipediag.diagnostics.likelihood_model import (
    GaussianLikelihoodModel,
)
from pipediag.diagnostics.likelihood_registry import (
    LikelihoodModelRegistry,
)


def create_model(mean_true=0.8, mean_false=0.2):
    return GaussianLikelihoodModel(
        mean_true=mean_true,
        std_true=0.1,
        mean_false=mean_false,
        std_false=0.1,
    )


def test_registry_stores_model():
    registry = LikelihoodModelRegistry()
    model = create_model()

    registry.register(
        evidence_group="G2",
        hypothesis="H2",
        model=model,
    )

    assert registry.get(
        evidence_group="G2",
        hypothesis="H2",
    ) is model


def test_registry_retrieves_correct_model_for_multiple_entries():
    registry = LikelihoodModelRegistry()

    model_h2 = create_model(mean_true=0.8)
    model_h7 = create_model(mean_true=0.3)

    registry.register("G2", "H2", model_h2)
    registry.register("G2", "H7", model_h7)

    assert registry.get("G2", "H2") is model_h2
    assert registry.get("G2", "H7") is model_h7


def test_different_evidence_groups_can_have_same_hypothesis():
    registry = LikelihoodModelRegistry()

    model_g2 = create_model(mean_true=0.8)
    model_g3 = create_model(mean_true=0.6)

    registry.register("G2", "H2", model_g2)
    registry.register("G3", "H2", model_g3)

    assert registry.get("G2", "H2") is model_g2
    assert registry.get("G3", "H2") is model_g3


def test_get_unknown_model_returns_none():
    registry = LikelihoodModelRegistry()

    assert registry.get(
        evidence_group="G2",
        hypothesis="H2",
    ) is None


def test_register_empty_evidence_group_is_rejected():
    registry = LikelihoodModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            evidence_group="",
            hypothesis="H2",
            model=create_model(),
        )


def test_register_empty_hypothesis_is_rejected():
    registry = LikelihoodModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            evidence_group="G2",
            hypothesis="",
            model=create_model(),
        )


def test_register_none_model_is_rejected():
    registry = LikelihoodModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            evidence_group="G2",
            hypothesis="H2",
            model=None,
        )


def test_duplicate_registration_is_rejected():
    registry = LikelihoodModelRegistry()

    registry.register(
        evidence_group="G2",
        hypothesis="H2",
        model=create_model(),
    )

    with pytest.raises(ValueError):
        registry.register(
            evidence_group="G2",
            hypothesis="H2",
            model=create_model(),
        )


def test_contains_returns_true_for_registered_model():
    registry = LikelihoodModelRegistry()

    registry.register(
        evidence_group="G2",
        hypothesis="H2",
        model=create_model(),
    )

    assert registry.contains("G2", "H2")


def test_contains_returns_false_for_unknown_model():
    registry = LikelihoodModelRegistry()

    assert not registry.contains("G2", "H2")


def test_registry_returns_all_models():
    registry = LikelihoodModelRegistry()

    model_h2 = create_model(mean_true=0.8)
    model_h7 = create_model(mean_true=0.3)

    registry.register("G2", "H2", model_h2)
    registry.register("G2", "H7", model_h7)

    result = registry.models()

    assert result == {
        ("G2", "H2"): model_h2,
        ("G2", "H7"): model_h7,
    }


def test_models_returns_copy_of_registry_state():
    registry = LikelihoodModelRegistry()

    model = create_model()

    registry.register(
        evidence_group="G2",
        hypothesis="H2",
        model=model,
    )

    result = registry.models()
    result[("G3", "H3")] = create_model()

    assert not registry.contains("G3", "H3")