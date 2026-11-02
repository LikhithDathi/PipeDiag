from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .context import RunContext
from .instrumentation import instrument_pipeline


@dataclass
class DiagnosticRun:
    """Public context manager for one PipeDiag execution.

    The run owns the runtime context and provides the public entry points
    needed by application code. Diagnostic inference remains delegated to a
    configured DiagnosticApplication.
    """

    context: RunContext
    application: Any | None = None
    metadata: dict[str, Any] | None = None

    def __enter__(self) -> "DiagnosticRun":
        self._started_at = datetime.now(timezone.utc)
        self._result = None
        self._closed = False
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._closed = True
        self._ended_at = datetime.now(timezone.utc)

        # Do not hide exceptions raised by the user's pipeline.
        return False

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def instrument(self, pipeline: Any) -> Any:
        """Instrument a supported scikit-learn Pipeline and return it."""
        instrument_pipeline(pipeline, self.context)
        return pipeline

    def result(
        self,
        *,
        application: Any | None = None,
        source: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Run the configured application over this execution context.

        ``application`` must be a configured DiagnosticApplication (or an
        object exposing ``run(source, metadata=...)``). The runtime-only API
        intentionally does not invent diagnostic calibration or evidence
        providers.
        """
        if (
            self._result is not None
            and application is None
            and source is None
            and metadata is None
        ):
            return self._result

        application = application or self.application

        if application is None:
            raise RuntimeError(
                "No diagnostic application is configured. "
                "Pass application=... to pipediag.start(...) "
                "or run.result(...)."
            )

        source = self.context if source is None else source

        merged_metadata = dict(self.metadata or {})

        if metadata:
            merged_metadata.update(metadata)

        self._result = application.run(
            source,
            metadata=merged_metadata or None,
        )

        return self._result