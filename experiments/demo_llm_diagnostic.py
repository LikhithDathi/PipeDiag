
from pathlib import Path

from experiments.generalization_evidence import (
    collect_generalization_evidence,
)
from experiments.overfitting_pipeline import run_overfitting_pipeline

from pipediag.application import DiagnosticApplication
from pipediag.diagnostics.engine_factory import (
    build_calibrated_diagnostic_engine,
)
from pipediag.llm.ollama import OllamaExplanationGenerator


class OverfittingEvidenceProvider:
    """Collect real G2 evidence from the executed pipeline."""

    def collect(self, source):
        return [collect_generalization_evidence(source)]


def main():
    print("\nRunning an intentionally overfit decision tree...")

    pipeline_result = run_overfitting_pipeline(
        n_samples=500,
        n_features=10,
        random_state=42,
        max_depth=None,
    )

    print("\nActual pipeline measurements")
    print("-" * 55)
    print(f"Training accuracy:     {pipeline_result.train_accuracy:.3f}")
    print(f"Evaluation accuracy:   {pipeline_result.test_accuracy:.3f}")
    print(f"Generalization gap:    {pipeline_result.generalization_gap:.3f}")

    application = DiagnosticApplication(
        evidence_provider=OverfittingEvidenceProvider(),
        diagnostic_engine=build_calibrated_diagnostic_engine(),
        explanation_generator=OllamaExplanationGenerator(
            model="phi3:latest",
        ),
    )

    print("\nRunning PipeDiag inference and LLM explanation...")

    result = application.run(pipeline_result)

    print("\n" + "=" * 70)
    print("PIPEDIAG — DIAGNOSTIC RESULTS")
    print("=" * 70)

    print("\nHypothesis posterior probabilities:")
    for hypothesis, probability in sorted(
        result.diagnostic_result.posterior.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {hypothesis}: {probability:.4f} ({probability:.1%})")

    print(
        f"\nDiagnostic uncertainty: "
        f"{result.diagnostic_result.uncertainty:.4f}"
    )
    print(
        f"Reporting threshold: "
        f"{result.explanation.threshold:.2f}"
    )

    if result.explanation.hypotheses:
        print("\nReported hypotheses:")
        for hypothesis in result.explanation.hypotheses:
            print(
                f"  {hypothesis.hypothesis_id} — "
                f"{hypothesis.title}: "
                f"{hypothesis.posterior:.1%}"
            )
    else:
        print("\nNo hypothesis crossed the reporting threshold.")

    print("\n" + "=" * 70)
    print("LLM-GENERATED USER REPORT")
    print("=" * 70)
    print(result.report)
    print("=" * 70)

    output_path = Path("artifacts/llm_overfitting_report.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.report, encoding="utf-8")

    print(f"\nReport saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
