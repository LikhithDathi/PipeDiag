from __future__ import annotations

from pipediag.diagnostics.adapters import diagnostic_evidence_to_group
from pipediag.diagnostics.evidence_group import EvidenceGroup
from pipediag.runtime.context import RunContext


def collect_evidence_groups(
    context: RunContext,
) -> list[EvidenceGroup]:
    """
    Convert evidence accumulated by a completed PipeDiag run
    into the grouped evidence representation used by diagnosis.
    """
    if context is None:
        raise ValueError("context is required")

    evidence = context.evidence_collector.get_evidence()

    return [
        diagnostic_evidence_to_group(item)
        for item in evidence
    ]