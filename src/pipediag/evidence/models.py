from dataclasses import dataclass, field
from typing import Any


FEATURE_DATA = "FEATURE_DATA"
TARGET_DATA = "TARGET_DATA"
PREDICTION_DATA = "PREDICTION_DATA"
UNKNOWN = "UNKNOWN"


@dataclass
class DatasetReference:
    """
    Lightweight identity and metadata for an observed data object.

    The framework stores metadata and identity information only.
    It does not store the underlying dataset.
    """

    dataset_id: str
    role: str

    shape: tuple[int, ...] | None = None
    dtype: str | None = None

    sample_count: int | None = None
    feature_count: int | None = None

    object_id: int | None = None

    source: str = "runtime"

    parent_dataset_ids: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)