from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceGroup:
    name: str
    value: Any
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("EvidenceGroup requires a non-empty name")

        if not self.status:
            raise ValueError("EvidenceGroup requires a non-empty status")