from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Event:
    """
    Represents one observable operation during a diagnostic run.
    """

    event_id: str
    run_id: str

    operation: str
    component: str | None = None
    step_name: str | None = None

    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float | None = None

    input_dataset_refs: list[str] = field(default_factory=list)
    output_dataset_refs: list[str] = field(default_factory=list)

    status: str = "running"

    parent_event_id: str | None = None
    depth: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)