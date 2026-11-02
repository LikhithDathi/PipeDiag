from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EVIDENCE_OBSERVED = "OBSERVED"
EVIDENCE_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass
class DiagnosticEvidence:
    """
    Common representation of evidence produced by a diagnostic.

    DiagnosticEvidence describes observed information and its
    availability. It does not make causal claims about the
    underlying ML pipeline.
    """

    evidence_type: str
    source: str

    dataset_ids: list[str] = field(
        default_factory=list
    )

    contexts: list[str] = field(
        default_factory=list
    )

    measurements: dict[str, Any] = field(
        default_factory=dict
    )

    status: str = EVIDENCE_OBSERVED

    reason: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )