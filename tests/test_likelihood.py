from pipediag.diagnostics.likelihood import LikelihoodObservation


def test_likelihood_observation_stores_required_information():
    observation = LikelihoodObservation(
        evidence_group="G2",
        hypothesis="H2",
        likelihood=0.8,
    )

    assert observation.evidence_group == "G2"
    assert observation.hypothesis == "H2"
    assert observation.likelihood == 0.8


def test_likelihood_observation_is_immutable():
    observation = LikelihoodObservation(
        evidence_group="G2",
        hypothesis="H2",
        likelihood=0.8,
    )

    try:
        observation.likelihood = 0.5
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "LikelihoodObservation should be immutable"
        )


def test_likelihood_must_be_between_zero_and_one():
    try:
        LikelihoodObservation(
            evidence_group="G2",
            hypothesis="H2",
            likelihood=1.2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Likelihood must be between 0 and 1"
        )


def test_negative_likelihood_is_rejected():
    try:
        LikelihoodObservation(
            evidence_group="G2",
            hypothesis="H2",
            likelihood=-0.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Likelihood must be between 0 and 1"
        )


def test_zero_and_one_are_valid_likelihood_values():
    zero = LikelihoodObservation(
        evidence_group="G2",
        hypothesis="H2",
        likelihood=0.0,
    )

    one = LikelihoodObservation(
        evidence_group="G2",
        hypothesis="H2",
        likelihood=1.0,
    )

    assert zero.likelihood == 0.0
    assert one.likelihood == 1.0