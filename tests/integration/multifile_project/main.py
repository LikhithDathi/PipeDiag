import pipediag

from .data.loader import load_data
from .evaluation.evaluator import evaluate
from .models.trainer import train_pipeline


def run_project():
    with pipediag.start() as run:
        X_train, X_test, y_train, y_test, split_evidence = load_data(
            run.context
        )

        pipeline = train_pipeline(
            X_train,
            y_train,
            run,
        )

        score = evaluate(
            pipeline,
            X_test,
            y_test,
        )

    return run, score, split_evidence