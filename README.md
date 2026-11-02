# PipeDiag

**An explainable diagnostic framework for scikit-learn machine learning pipelines.**

PipeDiag observes supported scikit-learn pipeline execution, records runtime events and dataset references, and provides components for evidence-based diagnosis, Bayesian inference, structured explanations, and human-readable reports.

The framework separates runtime observation from diagnostic inference so that applications can configure how evidence is collected and how diagnostic results are produced.

## Features

- **Runtime instrumentation:** Observe operations performed by supported scikit-learn pipelines.
- **Execution events:** Record component names, operations, event status, execution duration, dataset references, and event relationships.
- **Evidence-based diagnosis:** Evaluate diagnostic evidence against calibrated likelihood models.
- **Bayesian inference:** Produce a posterior distribution over diagnostic hypotheses.
- **Explainable results:** Build structured explanations and render diagnostic reports.
- **Configurable architecture:** Supply an evidence provider and diagnostic engine through the application layer.
- **Packaged calibration data:** Include the default likelihood-model artifact with the Python package.
- **Optional web dashboard:** Explore diagnostic runs through a locally hosted React dashboard.
- **Optional LLM integration:** Support Ollama-related explanation workflows through an optional dependency.

## Requirements

- Python 3.12 or later
- NumPy 2.0 or later
- scikit-learn 1.8 or later

## Installation

### Install from PyPI

Install the core PipeDiag package:

```bash
python -m pip install pipediag
```

The core installation includes runtime instrumentation, evidence-processing components, calibrated Bayesian diagnosis, explanation generation, and reporting functionality.

### Optional dependencies

PipeDiag provides optional integrations that can be installed separately.

**Web dashboard:**

```bash
python -m pip install "pipediag[web]"
```

Launch the dashboard:

```bash
pipediag dashboard
```

The dashboard is available at:

```text
http://127.0.0.1:8000
```

To customize the host and port:

```bash
pipediag dashboard --host 127.0.0.1 --port 8000 --no-browser
```

The dashboard runs locally and provides access to the PipeDiag backend and React frontend.

**Ollama integration:**

```bash
python -m pip install "pipediag[llm]"
```

**Install both integrations:**

```bash
python -m pip install "pipediag[web,llm]"
```

The `llm` extra installs Ollama-related support. The `web` extra installs FastAPI and Uvicorn. These extras are optional and are not part of the base dependency set.

### Instgit clone https://github.com/LikhithDathi/PipeDiag.git
cd PipeDiag
repository and install the package:

```bash
git clone <REPOSITORY_URL>
cd pipediag
python -m pip install .
```

For development, where source changes should be available without reinstalling:

```bash
python -m pip install -e .
```

### Install from a built wheel

If you have downloaded or built a wheel, install it with:

```bash
python -m pip install pipediag-0.1.0-py3-none-any.whl
```

Replace the filename with the wheel you have.

## Quick start: end-to-end pipeline diagnosis

PipeDiag provides the public `pipediag.start()` API for observing supported scikit-learn pipeline execution and connecting the captured runtime context to a configured diagnostic application.

### 1. Configure the diagnostic application

The diagnostic application coordinates evidence acquisition, Bayesian inference, explanation generation, and report rendering.

```python
from pipediag.application import (
    DiagnosticApplication,
    DiagnosticEvidenceProvider,
)

from pipediag.diagnostics.engine_factory import (
    build_calibrated_diagnostic_engine,
)

from pipediag.evidence.runtime_provider import (
    RuntimeEvidenceProvider,
)

application = DiagnosticApplication(
    evidence_provider=DiagnosticEvidenceProvider(
        RuntimeEvidenceProvider()
    ),
    diagnostic_engine=build_calibrated_diagnostic_engine(),
)
```

### 2. Instrument and execute a pipeline

```python
import numpy as np
import pipediag

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0],
    [4.0, 3.0],
    [5.0, 6.0],
    [6.0, 5.0],
    [7.0, 8.0],
    [8.0, 7.0],
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression()),
])

with pipediag.start(application=application) as run:
    pipeline = run.instrument(pipeline)

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    result = run.result()
```

### 3. Inspect the diagnostic result

```python
diagnostic = result.diagnostic_result

print("Run ID:", run.run_id)
print("Runtime events:", run.context.event_count())
print("Predictions:", predictions)

print("Posterior:", diagnostic.posterior)
print("Uncertainty:", diagnostic.uncertainty)

print("\nDiagnostic report:")
print(result.report)
```

The diagnostic result exposes:

- `posterior`: probability distribution over diagnostic hypotheses.
- `uncertainty`: uncertainty associated with the posterior distribution.
- `evidence`: evidence groups supplied to the diagnostic engine.
- `metadata`: contextual information associated with the diagnosis.

### What happens during a run?

1. `pipediag.start()` creates a `DiagnosticRun` and its `RunContext`.
2. `run.instrument(pipeline)` instruments the supported pipeline.
3. Pipeline operations execute normally while runtime events are recorded.
4. `RuntimeEvidenceProvider` acquires execution evidence from the completed runtime context.
5. `DiagnosticEvidenceProvider` adapts the acquired evidence into `EvidenceGroup` objects.
6. The calibrated diagnostic engine evaluates the evidence and performs Bayesian inference.
7. The application builds a structured explanation and renders a diagnostic report.

The context manager does not suppress exceptions raised by the user's pipeline.

### Current runtime evidence coverage

The current `RuntimeEvidenceProvider` implementation acquires execution evidence (G8) from captured runtime events. It calculates the execution failure rate, total event count, and failed event count.

The provider also contains a generalization-evidence helper, but that helper is not currently invoked by `collect()`. Other evidence groups are not automatically acquired by this provider.

Consequently, the end-to-end example demonstrates the configured diagnostic workflow using the currently implemented G8 runtime evidence. It should not be interpreted as automatic acquisition of all eight evidence groups.

## Runtime observation versus diagnostic inference

PipeDiag separates runtime observation from diagnostic inference. These capabilities can be used independently or combined through the public `DiagnosticRun` API.

### Runtime observation only

Use this mode when you want to instrument a pipeline and inspect its execution without performing Bayesian diagnosis.

```python
with pipediag.start() as run:
    run.instrument(pipeline)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

events = run.context.get_events()

for event in events:
    print(
        event.event_id,
        event.operation,
        event.status,
        event.duration_ms,
    )
```

This mode does not require a diagnostic application.

### End-to-end diagnosis

Use this mode when you want PipeDiag to acquire evidence and produce a diagnostic result.

Configure a `DiagnosticApplication` with:

- An evidence provider responsible for acquiring diagnostic evidence.
- A diagnostic engine responsible for Bayesian inference.

Pass the configured application to `pipediag.start()` and call `run.result()` after pipeline execution.

```python
with pipediag.start(application=application) as run:
    run.instrument(pipeline)
    pipeline.fit(X, y)

    result = run.result()
```

Alternatively, the application can be supplied directly to `run.result(application=application)`.

Calling `run.result()` without a configured application raises a `RuntimeError`.

### Diagnostic result structure

The application returns a `DiagnosticRunResult` containing:

- `diagnostic_result`: posterior distribution, uncertainty, evidence groups, and metadata.
- `explanation`: structured diagnostic interpretation.
- `report`: rendered human-readable report.

The posterior represents probabilistic diagnostic support. It should not be interpreted as proof of a causal root cause.

## Building a calibrated diagnostic engine

PipeDiag provides a factory for constructing a diagnostic engine from the packaged calibration artifact.

```python
from pipediag.diagnostics.engine_factory import (
    build_calibrated_diagnostic_engine,
)

engine = build_calibrated_diagnostic_engine()
```

The default calibration loader looks for the JSON artifact inside the installed package first. It also supports source-tree and shared-data fallback paths.

The loader validates the artifact schema, declared model count, required model fields, and duplicate model keys before returning a likelihood-model registry.

The calibrated engine can be used by a configured `DiagnosticApplication` or with the lower-level evidence interface when you already have compatible evidence groups.

## Working with diagnostic results

The lower-level diagnostic engine accepts evidence groups:

```python
from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.diagnostics.engine_factory import (
    build_calibrated_diagnostic_engine,
)

engine = build_calibrated_diagnostic_engine()

evidence = [
    EvidenceGroup(
        name="G1",
        value=0.8,
        status="observed",
    )
]

result = engine.diagnose(evidence)

print("Posterior:", result.posterior)
print("Uncertainty:", result.uncertainty)
```

This is a minimal API example, not a validated real-world diagnostic scenario. The meaning and validity of evidence values depend on the evidence-generation process and the calibration models. Do not interpret a single synthetic value as proof of a pipeline defect.

A diagnostic result includes:

- `posterior`: probabilities associated with diagnostic hypotheses.
- `uncertainty`: the engine's uncertainty measure.
- `evidence`: the evidence groups supplied to the engine.
- `metadata`: optional contextual information.

The current hypothesis set includes data leakage, generalization failure, distribution shift, class imbalance, data-quality problems, evaluation problems, underfitting, feature-related problems, and pipeline/execution problems.

## Calibration data

The package includes a default artifact at:

```text
pipediag/calibration/likelihood_models.json
```

The artifact uses schema version 1 and contains calibrated likelihood-model entries for the diagnostic engine.

The default loader resolves the artifact from the installed package when available. It also supports fallback locations for source checkouts and shared data installations.

If you supply a custom artifact path, ensure that the JSON follows the schema expected by the installed version of PipeDiag.

A successful calibration load establishes that the artifact can be parsed and used to construct the registry. It does not, by itself, establish that the resulting probabilities are reliable for every application or dataset. Calibration quality must be assessed against appropriate evaluation scenarios.

## Regenerating calibration artifacts

PipeDiag provides a script for regenerating the calibrated likelihood-model artifact.

The calibration models are generated using the calibration workflow in `experiments/full_system_validation.py`.

To regenerate the artifacts, run the following command from the project root:

```bash
python scripts/export_calibration.py
```

The script:

1. Builds the calibrated likelihood-model registry.
2. Extracts the likelihood parameters for each model.
3. Validates that exactly 11 models have been generated.
4. Serializes the models into the versioned JSON artifact format.
5. Writes the artifact to both calibration locations.
6. Reloads and verifies both generated JSON files.

The generated artifacts are stored at:

```text
artifacts/calibration/likelihood_models.json
src/pipediag/calibration/likelihood_models.json
```

The package-level artifact under `src/pipediag/calibration/` is tracked in the repository and included in the Python distribution.

The repository-level artifact under `artifacts/calibration/` is a generated development artifact. It is excluded from Git and can be regenerated by running the export script.

Both files should remain synchronized when the export script is run. The script updates and verifies both locations automatically.

The calibration artifact uses schema version 1 and currently contains 11 likelihood models.

Regenerating the artifact does not itself establish the scientific validity of the calibration methodology or its underlying data. Those require separate validation.

## Architecture overview

PipeDiag separates runtime observation from inference and presentation.

```text
Scikit-learn pipeline
        |
        v
Runtime instrumentation
        |
        v
RunContext and execution events
        |
        v
Configured evidence provider
        |
        v
Evidence groups
        |
        v
Calibrated diagnostic engine
        |
        v
Bayesian posterior and uncertainty
        |
        v
Structured explanation
        |
        v
Human-readable report
```

The diagram describes the high-level application workflow. Runtime events are not automatically equivalent to diagnostic evidence; a configured provider is responsible for obtaining and adapting the evidence required by the diagnostic engine.

## Web dashboard

PipeDiag includes an optional local web dashboard built with React and powered by a FastAPI backend.

Install the web dependencies:

```bash
python -m pip install "pipediag[web]"
```

Launch the dashboard:

```bash
pipediag dashboard
```

By default, the dashboard is served at:

```text
http://127.0.0.1:8000
```

The dashboard provides an interface for exploring diagnostic runs, viewing diagnostic results, and accessing saved run history.

The backend exposes API endpoints for health checks, diagnostic runs, and example execution.

To stop the dashboard, press `Ctrl+C` in the terminal running the server.

The dashboard is optional. Users who only need the Python diagnostic framework do not need to install the web dependencies.

## Development and tests

To install the project for development:

```bash
python -m pip install -e .
```

Install the optional web dependencies when working on the dashboard:

```bash
python -m pip install -e ".[web]"
```

Install the optional LLM dependencies when working on the Ollama integration:

```bash
python -m pip install -e ".[llm]"
```

Run the test suite with:

```bash
python -m pytest -q
```

Build the distribution artifacts with:

```bash
python -m pip install build
python -m build
```

The wheel and source distribution are written to the `dist/` directory.

For release verification, install the built wheel in a fresh virtual environment and check both runtime instrumentation and calibration loading.

## Scope and limitations

- Runtime instrumentation targets supported scikit-learn pipelines; it is not a general-purpose Python execution tracer.
- Runtime events and dataset references provide execution evidence, but they do not independently establish the semantic meaning of every feature or the root cause of a model problem.
- Full diagnostic inference requires a compatible evidence provider and configured diagnostic engine.
- The current runtime evidence provider automatically acquires execution evidence (G8); other evidence groups require their own acquisition and integration.
- Diagnostic outputs depend on the quality, coverage, and applicability of the evidence and calibration models.
- A confident posterior should not be interpreted as a guarantee of correctness without appropriate empirical validation.
- Optional integrations may require additional dependencies and their own configuration.

## Project status

PipeDiag includes runtime instrumentation, evidence-processing components, calibrated Bayesian diagnosis, explanation and reporting components, and optional integration support.

The project is under active development. Before using it in a production decision-making workflow, validate the complete diagnostic process against representative scenarios, verify probability calibration, and document the supported deployment configuration.

## License

PipeDiag is released under the MIT License.

Copyright (c) 2026 Dathi Likhith.

See the [LICENSE](LICENSE) file for details.