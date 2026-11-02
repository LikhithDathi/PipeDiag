from .runtime import DiagnosticRun, Event, RunContext
from .persistence import (
    delete_run,
    list_runs,
    load_run,
    save_run,
)


def start(*, application=None, metadata=None) -> DiagnosticRun:
    """Start a PipeDiag diagnostic run."""
    return DiagnosticRun(
        context=RunContext(),
        application=application,
        metadata=metadata,
    )


__all__ = [
    "DiagnosticRun",
    "Event",
    "RunContext",
    "start",
    "save_run",
    "list_runs",
    "load_run",
    "delete_run",
]