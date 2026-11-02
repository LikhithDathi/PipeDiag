import math

from experiments.missing_evidence_validation import (
    run_missing_evidence_validation,
    run_multiple_missing_evidence_validation,
    uniform_prior,
)


HYPOTHESIS_IDS = {
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


def _assert_valid_posterior(posterior):
    assert set(posterior) == HYPOTHESIS_IDS

    assert all(
        math.isfinite(probability)
        and 0.0 <= probability <= 1.0
        for probability in posterior.values()
    )

    assert math.isclose(
        sum(posterior.values()),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_complete_evidence_produces_valid_posterior():
    validation = run_missing_evidence_validation()

    _assert_valid_posterior(validation.complete_posterior)


def test_complete_h3_evidence_supports_h3():
    validation = run_missing_evidence_validation()

    assert validation.complete_posterior["H3"] > 0.5


def test_removing_g3_does_not_treat_absence_as_negative_evidence():
    validation = run_missing_evidence_validation()

    prior = uniform_prior()

    assert validation.missing_posterior["H3"] >= prior["H3"]


def test_missing_evidence_preserves_h3_support_property():
    validation = run_missing_evidence_validation()

    assert validation.missing_evidence_preserves_h3_support is True


def test_missing_evidence_produces_valid_posterior():
    validation = run_missing_evidence_validation()

    _assert_valid_posterior(validation.missing_posterior)


def test_missing_evidence_does_not_produce_invalid_uncertainty():
    validation = run_missing_evidence_validation()

    assert math.isfinite(validation.missing_uncertainty)
    assert 0.0 <= validation.missing_uncertainty <= 1.0


def test_no_evidence_returns_prior_distribution():
    validation = run_missing_evidence_validation()

    prior = uniform_prior()

    for hypothesis in prior:
        assert math.isclose(
            validation.no_evidence_posterior[hypothesis],
            prior[hypothesis],
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


def test_no_evidence_matches_prior_property():
    validation = run_missing_evidence_validation()

    assert validation.no_evidence_matches_prior is True


def test_no_evidence_has_maximum_uncertainty():
    validation = run_missing_evidence_validation()

    assert math.isclose(
        validation.no_evidence_uncertainty,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_no_evidence_has_maximum_uncertainty_property():
    validation = run_missing_evidence_validation()

    assert validation.no_evidence_has_maximum_uncertainty is True


def test_multiple_missing_evidence_groups_produce_valid_posterior():
    validation = run_multiple_missing_evidence_validation()

    _assert_valid_posterior(validation.missing_posterior)


def test_multiple_missing_evidence_groups_do_not_create_negative_evidence():
    validation = run_multiple_missing_evidence_validation()

    prior = uniform_prior()

    assert validation.missing_posterior["H3"] >= prior["H3"]


def test_multiple_missing_evidence_groups_have_finite_uncertainty():
    validation = run_multiple_missing_evidence_validation()

    assert math.isfinite(validation.missing_uncertainty)
    assert 0.0 <= validation.missing_uncertainty <= 1.0


def test_missing_evidence_validation_passes():
    validation = run_missing_evidence_validation()

    assert validation.passed is True


def test_multiple_missing_evidence_validation_passes():
    validation = run_multiple_missing_evidence_validation()

    assert validation.passed is True