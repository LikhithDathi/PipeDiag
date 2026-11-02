from dataclasses import dataclass


@dataclass(frozen=True)
class Hypothesis:
    id: str
    name: str
    description: str


DIAGNOSTIC_HYPOTHESES = (
    Hypothesis(
        id="H1",
        name="Data Leakage",
        description=(
            "Information may have crossed the intended "
            "training/evaluation information boundary."
        ),
    ),
    Hypothesis(
        id="H2",
        name="Generalization Failure",
        description=(
            "Model performance degrades substantially on unseen "
            "evaluation data relative to training/reference data."
        ),
    ),
    Hypothesis(
        id="H3",
        name="Distribution Shift",
        description=(
            "Evaluation/reference data distributions differ materially "
            "from the reference/training distribution."
        ),
    ),
    Hypothesis(
        id="H4",
        name="Class Imbalance",
        description=(
            "Unequal class representation is a plausible contributor "
            "to the observed model problem."
        ),
    ),
    Hypothesis(
        id="H5",
        name="Data Quality Problem",
        description=(
            "Observable data-integrity or data-quality issues may "
            "contribute to the observed problem."
        ),
    ),
    Hypothesis(
        id="H6",
        name="Evaluation Problem",
        description=(
            "The evaluation setup or available evaluation evidence "
            "is insufficient or unreliable for judging model performance."
        ),
    ),
    Hypothesis(
        id="H7",
        name="Underfitting",
        description=(
            "The model fails to capture sufficient predictive structure, "
            "resulting in poor training and evaluation performance."
        ),
    ),
    Hypothesis(
        id="H8",
        name="Feature-related Problem",
        description=(
            "Measurable properties of the feature representation may "
            "limit model performance or reliability."
        ),
    ),
    Hypothesis(
        id="H9",
        name="Pipeline/Execution Problem",
        description=(
            "Observable runtime execution anomalies may affect "
            "the pipeline result."
        ),
    ),
)