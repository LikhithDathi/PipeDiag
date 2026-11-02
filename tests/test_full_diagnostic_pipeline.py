import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipediag.diagnostics.adapters import (
    diagnostic_evidence_to_group,
    generalization_to_evidence,
)
from pipediag.diagnostics.bayesian import BayesianInference
from pipediag.diagnostics.bayesian_evidence import BayesianEvidenceInference
from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)
from pipediag.diagnostics.evidence_likelihood import (
    EvidenceLikelihoodEvaluator,
)
from pipediag.diagnostics.generalization import (
    HIGHER_IS_BETTER,
    GeneralizationDiagnostic,
)
from pipediag.diagnostics.likelihood_model import GaussianLikelihoodModel
from pipediag.diagnostics.likelihood_registry import LikelihoodModelRegistry
from pipediag.runtime.context import RunContext
from pipediag.runtime.instrumentation import instrument_pipeline
from pipediag.evidence import (
    FEATURE_DATA,
    TARGET_DATA,
)
from pipediag.evidence.store import (
    TRAIN,
    TEST,
)


HYPOTHESES = [f"H{i}" for i in range(1, 10)]


def make_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=500),
            ),
        ]
    )


def make_likelihood_registry():
    registry = LikelihoodModelRegistry()

    registry.register(
        "G2",
        "H2",
        GaussianLikelihoodModel(
            mean_true=0.30,
            std_true=0.10,
            mean_false=0.02,
            std_false=0.05,
            measurement="gap",
        ),
    )

    registry.register(
        "G2",
        "H7",
        GaussianLikelihoodModel(
            mean_true=0.05,
            std_true=0.05,
            mean_false=0.20,
            std_false=0.10,
            measurement="gap",
        ),
    )

    return registry


def test_real_pipeline_reaches_bayesian_diagnostic():
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    pipeline = make_pipeline()

    context = RunContext()

    instrument_pipeline(
        pipeline,
        context,
    )

    # --------------------------------------------------------------
    # Train the real pipeline.
    # --------------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------------
    # Register the actual dataset objects with the collector.
    #
    # The collector creates stable DatasetReferences for these
    # objects. Their contexts are then explicitly assigned rather
    # than inferred from the fact that predict/score was called.
    # --------------------------------------------------------------

    train_x_ref = context.evidence_collector._reference_for(
        X_train,
        role=FEATURE_DATA,
    )

    train_y_ref = context.evidence_collector._reference_for(
        y_train,
        role=TARGET_DATA,
    )

    test_x_ref = context.evidence_collector._reference_for(
        X_test,
        role=FEATURE_DATA,
    )

    test_y_ref = context.evidence_collector._reference_for(
        y_test,
        role=TARGET_DATA,
    )

    context.dataset_contexts.set_context(
        train_x_ref.dataset_id,
        TRAIN,
    )

    context.dataset_contexts.set_context(
        train_y_ref.dataset_id,
        TRAIN,
    )

    context.dataset_contexts.set_context(
        test_x_ref.dataset_id,
        TEST,
    )

    context.dataset_contexts.set_context(
        test_y_ref.dataset_id,
        TEST,
    )

    # --------------------------------------------------------------
    # Evaluate the real instrumented pipeline.
    #
    # score() is important here because the existing runtime
    # instrumentation sends score results through the
    # EvaluationAcquirer.
    # --------------------------------------------------------------

    pipeline.score(
        X_train,
        y_train,
    )

    pipeline.score(
        X_test,
        y_test,
    )

    # --------------------------------------------------------------
    # Evaluation evidence must now come from the runtime itself.
    # Nothing is manually constructed here.
    # --------------------------------------------------------------

    evaluations = context.get_evaluations()

    assert len(evaluations) == 2

    for evaluation in evaluations:
        print(
            "\nEVALUATION:",
            evaluation,
        )

    print(
        "\nDATASETS:"
    )

    for reference in (
        context.evidence_collector.lineage.get_all()
    ):
        print(
            reference.dataset_id,
            reference.role,
            reference.shape,
            reference.object_id,
        )

    assert all(
        evaluation.metric_value is not None
        for evaluation in evaluations
    )

    # --------------------------------------------------------------
    # Generalization diagnostic
    # --------------------------------------------------------------

    generalization = GeneralizationDiagnostic().analyze(
        evaluations,
        metric_direction=HIGHER_IS_BETTER,
    )

    assert generalization.status == "COMPUTED"

    assert generalization.train_value is not None
    assert generalization.test_value is not None
    assert generalization.gap is not None

    # --------------------------------------------------------------
    # Convert diagnostic evidence into the Bayesian evidence group.
    # --------------------------------------------------------------

    diagnostic_evidence = generalization_to_evidence(
        generalization,
    )

    assert diagnostic_evidence.evidence_type == "generalization"
    assert diagnostic_evidence.status == "OBSERVED"

    group = diagnostic_evidence_to_group(
        diagnostic_evidence,
    )

    assert group.name == "G2"
    assert group.status == "observed"

    assert group.metadata["measurements"]["gap"] == (
        generalization.gap
    )

    # --------------------------------------------------------------
    # Bayesian likelihood evaluation
    # --------------------------------------------------------------

    registry = make_likelihood_registry()

    evaluator = EvidenceLikelihoodEvaluator(
        registry=registry,
        evidence_hypothesis_map=EVIDENCE_HYPOTHESIS_MAP,
    )

    log_ratios = evaluator.evaluate_log_ratios(
        group,
    )

    assert set(log_ratios) == {
        "H2",
        "H7",
    }

    assert all(
        np.isfinite(value)
        for value in log_ratios.values()
    )

    # --------------------------------------------------------------
    # Bayesian inference
    # --------------------------------------------------------------

    bayesian = BayesianEvidenceInference(
        BayesianInference(HYPOTHESES),
    )

    posterior = bayesian.infer_with_log_likelihood_ratios(
        {
            group.name: log_ratios,
        }
    )

    assert set(posterior) == set(HYPOTHESES)

    assert np.isclose(
        sum(posterior.values()),
        1.0,
    )

    assert all(
        np.isfinite(value)
        and value >= 0
        for value in posterior.values()
    )

    # --------------------------------------------------------------
    # The observed evidence must affect the prior.
    #
    # With nine uniformly distributed hypotheses, the uninformed
    # prior is 1/9 for each hypothesis.
    #
    # G2 is directly associated with H2 and H7, so at least one
    # of those hypotheses must move away from the prior.
    # --------------------------------------------------------------

    prior = 1.0 / len(HYPOTHESES)

    assert (
        not np.isclose(
            posterior["H2"],
            prior,
        )
        or not np.isclose(
            posterior["H7"],
            prior,
        )
    )

    # --------------------------------------------------------------
    # Runtime observation must have occurred.
    # --------------------------------------------------------------

    assert context.event_count() > 0