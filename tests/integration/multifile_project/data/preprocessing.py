from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessing_steps():
    return [
        ("scaler", StandardScaler()),
    ]
