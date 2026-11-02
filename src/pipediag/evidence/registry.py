from __future__ import annotations

from typing import Any

from .models import DatasetReference, UNKNOWN


class DatasetRegistry:
    """
    Maps observed Python objects to DatasetReference instances.

    Objects are retained for the lifetime of the registry so that
    Python object IDs cannot be reused for a different object while
    the diagnostic run is still active.
    """

    def __init__(self) -> None:
        self._references: dict[int, DatasetReference] = {}
        self._objects: dict[int, Any] = {}

    def get(self, obj: Any) -> DatasetReference | None:
        """
        Return the existing reference for an object, if one exists.
        """
        return self._references.get(id(obj))


    def get_object(
        self,
        dataset_id: str,
    ) -> Any | None:
        """
        Return the retained Python object associated with a dataset ID.

        The registry retains objects for the lifetime of the diagnostic run.
        This method resolves a dataset ID through its DatasetReference rather
        than exposing the internal object storage directly.
        """
        for object_id, reference in self._references.items():
            if reference.dataset_id == dataset_id:
                return self._objects.get(object_id)

        return None



    def register(
        self,
        obj: Any,
        reference: DatasetReference,
    ) -> DatasetReference:
        """
        Associate an object with a dataset reference and retain the
        object for the lifetime of this registry.
        """

        object_id = id(obj)

        self._references[object_id] = reference
        self._objects[object_id] = obj

        return reference

    def get_or_create(
        self,
        obj: Any,
        create_reference,
    ) -> DatasetReference:

        existing = self.get(obj)

        if existing is not None:
            return existing

        reference = create_reference(
            obj=obj,
            role=UNKNOWN,
        )

        return self.register(obj, reference)


    @property
    def datasets(self) -> dict[str, DatasetReference]:
        """Return the registered dataset references keyed by dataset ID."""
        return {
            reference.dataset_id: reference
            for reference in self._references.values()
        }


    def count(self) -> int:
        return len(self._references)
