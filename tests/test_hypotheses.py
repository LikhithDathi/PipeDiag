from pipediag.diagnostics.hypotheses import (
    DIAGNOSTIC_HYPOTHESES,
    Hypothesis,
)


def test_hypothesis_collection_contains_nine_diagnostic_hypotheses():
    assert len(DIAGNOSTIC_HYPOTHESES) == 9


def test_hypothesis_ids_are_unique():
    ids = [hypothesis.id for hypothesis in DIAGNOSTIC_HYPOTHESES]

    assert len(ids) == len(set(ids))


def test_expected_hypothesis_ids_exist():
    ids = {hypothesis.id for hypothesis in DIAGNOSTIC_HYPOTHESES}

    assert ids == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
    }


def test_hypothesis_order_is_deterministic():
    assert [hypothesis.id for hypothesis in DIAGNOSTIC_HYPOTHESES] == [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
    ]


def test_each_hypothesis_has_required_information():
    for hypothesis in DIAGNOSTIC_HYPOTHESES:
        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.id
        assert hypothesis.name
        assert hypothesis.description


def test_hypothesis_names_are_correct():
    names = {
        hypothesis.id: hypothesis.name
        for hypothesis in DIAGNOSTIC_HYPOTHESES
    }

    assert names == {
        "H1": "Data Leakage",
        "H2": "Generalization Failure",
        "H3": "Distribution Shift",
        "H4": "Class Imbalance",
        "H5": "Data Quality Problem",
        "H6": "Evaluation Problem",
        "H7": "Underfitting",
        "H8": "Feature-related Problem",
        "H9": "Pipeline/Execution Problem",
    }


def test_hypotheses_are_immutable():
    hypothesis = DIAGNOSTIC_HYPOTHESES[0]

    try:
        hypothesis.name = "Changed"
    except AttributeError:
        pass
    else:
        raise AssertionError("Hypothesis should be immutable")