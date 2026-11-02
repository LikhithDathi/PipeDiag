from __future__ import annotations

from typing import Any

from .models import DatasetReference, UNKNOWN


class DatasetLineage:
    """
    Maintains lightweight dataset references and their relationships
    during a diagnostic run.
    """

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetReference] = {}
        self._counter = 0

    def create_reference(
        self,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
        role: str = UNKNOWN,
        object_id: int | None = None,
        parent_dataset_ids: list[str] | None = None,
        source: str = "runtime",
        metadata: dict[str, Any] | None = None,
    ) -> DatasetReference:

        self._counter += 1
        dataset_id = f"D-{self._counter:04d}"

        sample_count = None
        feature_count = None

        if shape:
            if len(shape) >= 1:
                sample_count = shape[0]

            if len(shape) >= 2:
                feature_count = shape[1]

        reference = DatasetReference(
            dataset_id=dataset_id,
            role=role,
            shape=shape,
            dtype=dtype,
            sample_count=sample_count,
            feature_count=feature_count,
            object_id=object_id,
            source=source,
            parent_dataset_ids=parent_dataset_ids or [],
            metadata=metadata or {},
        )

        self._datasets[dataset_id] = reference

        return reference

    def get(
        self,
        dataset_id: str,
    ) -> DatasetReference | None:
        return self._datasets.get(dataset_id)

    def get_all(self) -> list[DatasetReference]:
        return list(self._datasets.values())

    def count(self) -> int:
        return len(self._datasets)

    def parents_of(
        self,
        dataset_id: str,
    ) -> list[str]:

        reference = self.get(dataset_id)

        if reference is None:
            return []

        return list(reference.parent_dataset_ids)