from __future__ import annotations


TRAIN = "TRAIN"
VALIDATION = "VALIDATION"
TEST = "TEST"
UNKNOWN = "UNKNOWN"


class DatasetContextRegistry:
    """
    Stores the observed evaluation context of datasets.

    Dataset semantic role and dataset context are intentionally
    separate concepts.

    Example:
        D-001 -> FEATURE_DATA + TRAIN
        D-002 -> FEATURE_DATA + TEST
    """

    def __init__(self) -> None:
        self._contexts: dict[str, str] = {}

    def set_context(
        self,
        dataset_id: str,
        context: str,
    ) -> None:
        self._contexts[dataset_id] = context

    def get_context(
        self,
        dataset_id: str,
    ) -> str:
        return self._contexts.get(
            dataset_id,
            UNKNOWN,
        )

    def has_context(
        self,
        dataset_id: str,
    ) -> bool:
        return dataset_id in self._contexts