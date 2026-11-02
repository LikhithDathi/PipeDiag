import pytest

import pipediag

from pipediag.runtime import DiagnosticRun


def test_start_creates_public_run_context():
    run = pipediag.start()

    assert isinstance(run, DiagnosticRun)
    assert run.context is not None
    assert run.run_id == run.context.run_id


def test_run_instrument_instruments_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    with pipediag.start() as run:
        returned_pipeline = run.instrument(pipeline)

        assert returned_pipeline is pipeline

        pipeline.fit(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ],
            [0, 0, 1, 1],
        )

    events = run.context.get_events()

    assert len(events) > 0


def test_result_requires_diagnostic_application():
    with pipediag.start() as run:
        with pytest.raises(
            RuntimeError,
            match="No diagnostic application is configured",
        ):
            run.result()