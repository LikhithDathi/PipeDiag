from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..data.preprocessing import build_preprocessing_steps


def build_pipeline():
    return Pipeline(
        build_preprocessing_steps()
        + [
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            )
        ]
    )


def train_pipeline(X_train, y_train, run):
    pipeline = build_pipeline()

    run.instrument(pipeline)

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline