from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pipediag.evidence import (
    FEATURE_DATA,
    TARGET_DATA,
    UNKNOWN,
    DatasetLineage,
    DatasetReference,
    DatasetRegistry,
)

from pipediag.evidence.store import (
    DatasetContextRegistry,
    TRAIN,
    TEST,
)




@dataclass
class SplitConfiguration:
    test_size_requested: Any = None
    train_size_requested: Any = None
    shuffle_requested: bool = True
    random_state: Any = None
    stratify_requested: bool = False


@dataclass
class SplitResult:
    n_input: int | None = None
    n_train: int | None = None
    n_test: int | None = None

    effective_train_ratio: float | None = None
    effective_test_ratio: float | None = None


@dataclass
class RepresentativenessEvidence:
    status: str = "not_assessed"
    metric: str | None = None
    value: float | None = None

    overall_distribution: dict[Any, float] | None = None
    train_distribution: dict[Any, float] | None = None
    test_distribution: dict[Any, float] | None = None


@dataclass
class SplitEvidence:
    evidence_type: str = "data_split"
    status: str = "observed"
    method: str = "train_test_split"

    configuration: SplitConfiguration = field(
        default_factory=SplitConfiguration
    )

    result: SplitResult = field(
        default_factory=SplitResult
    )

    input_dataset_ids: list[str] = field(
        default_factory=list
    )

    output_dataset_ids: list[str] = field(
        default_factory=list
    )

    representativeness: RepresentativenessEvidence = field(
        default_factory=RepresentativenessEvidence
    )


class SplitAcquirer:
    """
    Acquires evidence from sklearn.model_selection.train_test_split.

    The original sklearn behavior is preserved.

    This component records:
        - split configuration
        - input datasets
        - output datasets
        - dataset lineage
        - train/test context
        - effective split ratios
        - safe representativeness statistics
    """

    def __init__(
        self,
        lineage: DatasetLineage | None = None,
        registry: DatasetRegistry | None = None,
        context_registry: DatasetContextRegistry | None = None,
    ) -> None:

        self.lineage = lineage or DatasetLineage()

        self.registry = (
            registry
            or DatasetRegistry()
        )

        self.context_registry = (
            context_registry
            or DatasetContextRegistry()
        )

    def split(
        self,
        *arrays,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    ):
        from sklearn.model_selection import train_test_split

        configuration = SplitConfiguration(
            test_size_requested=test_size,
            train_size_requested=train_size,
            shuffle_requested=shuffle,
            random_state=random_state,
            stratify_requested=stratify is not None,
        )

        input_references = self._register_inputs(
            arrays,
        )

        outputs = train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        output_references = self._register_outputs(
            arrays,
            outputs,
            input_references,
        )

        self._assign_split_contexts(
            output_references,
            len(arrays),
        )

        result = self._build_result(
            arrays,
            outputs,
        )

        representativeness = (
            self._assess_representativeness(
                arrays,
                outputs,
            )
        )

        evidence = SplitEvidence(
            configuration=configuration,
            result=result,
            input_dataset_ids=[
                reference.dataset_id
                for reference in input_references
            ],
            output_dataset_ids=[
                reference.dataset_id
                for reference in output_references
            ],
            representativeness=representativeness,
        )

        return outputs, evidence

    # ------------------------------------------------------------------
    # Input registration
    # ------------------------------------------------------------------

    def _register_inputs(
        self,
        arrays,
    ) -> list[DatasetReference]:

        references = []

        for index, array in enumerate(arrays):

            if index == 0:
                role = FEATURE_DATA

            elif index == 1:
                role = TARGET_DATA

            else:
                role = UNKNOWN

            references.append(
                self._reference_for(
                    array,
                    role=role,
                )
            )

        return references

    # ------------------------------------------------------------------
    # Output registration
    # ------------------------------------------------------------------

    def _register_outputs(
        self,
        inputs,
        outputs,
        input_references,
    ) -> list[DatasetReference]:

        references = []

        n_arrays = len(inputs)

        for index in range(n_arrays):

            train_output = outputs[
                2 * index
            ]

            test_output = outputs[
                2 * index + 1
            ]

            if index == 0:
                role = FEATURE_DATA

            elif index == 1:
                role = TARGET_DATA

            else:
                role = UNKNOWN

            parent_id = (
                input_references[index].dataset_id
            )

            train_reference = self._reference_for(
                train_output,
                role=role,
                parent_dataset_ids=[
                    parent_id
                ],
            )

            test_reference = self._reference_for(
                test_output,
                role=role,
                parent_dataset_ids=[
                    parent_id
                ],
            )

            references.extend(
                [
                    train_reference,
                    test_reference,
                ]
            )

        return references

    # ------------------------------------------------------------------
    # Train / Test context
    # ------------------------------------------------------------------

    def _assign_split_contexts(
        self,
        output_references: list[DatasetReference],
        n_arrays: int,
    ) -> None:

        for index in range(n_arrays):

            train_reference = output_references[
                2 * index
            ]

            test_reference = output_references[
                2 * index + 1
            ]

            self.context_registry.set_context(
                train_reference.dataset_id,
                TRAIN,
            )

            self.context_registry.set_context(
                test_reference.dataset_id,
                TEST,
            )

    # ------------------------------------------------------------------
    # Result information
    # ------------------------------------------------------------------

    def _build_result(
        self,
        inputs,
        outputs,
    ) -> SplitResult:

        if not inputs:
            return SplitResult()

        n_input = self._sample_count(
            inputs[0]
        )

        n_train = self._sample_count(
            outputs[0]
        )

        n_test = self._sample_count(
            outputs[1]
        )

        train_ratio = None
        test_ratio = None

        if n_input:
            train_ratio = n_train / n_input
            test_ratio = n_test / n_input

        return SplitResult(
            n_input=n_input,
            n_train=n_train,
            n_test=n_test,
            effective_train_ratio=train_ratio,
            effective_test_ratio=test_ratio,
        )

    # ------------------------------------------------------------------
    # Representativeness
    # ------------------------------------------------------------------

    def _assess_representativeness(
        self,
        inputs,
        outputs,
    ) -> RepresentativenessEvidence:

        if len(inputs) < 2:
            return RepresentativenessEvidence(
                status="not_assessed",
            )

        y = inputs[1]

        try:
            y_train = outputs[2]
            y_test = outputs[3]

        except IndexError:
            return RepresentativenessEvidence(
                status="insufficient_data",
            )

        overall = self._distribution(y)
        train = self._distribution(y_train)
        test = self._distribution(y_test)

        if not overall or not train or not test:
            return RepresentativenessEvidence(
                status="insufficient_data",
            )

        classes = (
            set(overall)
            | set(train)
            | set(test)
        )

        # Binary case
        if len(classes) == 2:

            positive_class = sorted(
                classes,
                key=lambda value: str(value),
            )[-1]

            value = abs(
                train.get(
                    positive_class,
                    0.0,
                )
                - test.get(
                    positive_class,
                    0.0,
                )
            )

            return RepresentativenessEvidence(
                status="assessed",
                metric=(
                    "binary_distribution_difference"
                ),
                value=float(value),
                overall_distribution=overall,
                train_distribution=train,
                test_distribution=test,
            )

        # Multiclass case
        value = self._total_variation_distance(
            train,
            test,
            classes,
        )

        return RepresentativenessEvidence(
            status="assessed",
            metric="total_variation_distance",
            value=float(value),
            overall_distribution=overall,
            train_distribution=train,
            test_distribution=test,
        )

    @staticmethod
    def _distribution(
        values,
    ) -> dict[Any, float] | None:

        try:
            array = np.asarray(values)

            if (
                array.ndim == 0
                or len(array) == 0
            ):
                return None

            unique, counts = np.unique(
                array,
                return_counts=True,
            )

            total = counts.sum()

            if total == 0:
                return None

            return {
                (
                    value.item()
                    if hasattr(value, "item")
                    else value
                ): float(count / total)
                for value, count in zip(
                    unique,
                    counts,
                )
            }

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _total_variation_distance(
        first: dict[Any, float],
        second: dict[Any, float],
        classes: set[Any],
    ) -> float:

        return 0.5 * sum(
            abs(
                first.get(
                    label,
                    0.0,
                )
                - second.get(
                    label,
                    0.0,
                )
            )
            for label in classes
        )

    # ------------------------------------------------------------------
    # Dataset reference helpers
    # ------------------------------------------------------------------

    def _reference_for(
        self,
        obj,
        role,
        parent_dataset_ids=None,
    ) -> DatasetReference:

        existing = self.registry.get(obj)

        if existing is not None:
            return existing

        shape = self._extract_shape(obj)
        dtype = self._extract_dtype(obj)

        reference = self.lineage.create_reference(
            shape=shape,
            dtype=dtype,
            role=role,
            object_id=id(obj),
            parent_dataset_ids=(
                parent_dataset_ids or []
            ),
        )

        self.registry.register(
            obj,
            reference,
        )

        return reference

    @staticmethod
    def _extract_shape(obj):

        shape = getattr(
            obj,
            "shape",
            None,
        )

        if shape is None:

            try:
                return (len(obj),)

            except TypeError:
                return None

        try:
            return tuple(
                int(value)
                for value in shape
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _extract_dtype(obj):

        dtype = getattr(
            obj,
            "dtype",
            None,
        )

        if dtype is None:
            return None

        return str(dtype)

    @staticmethod
    def _sample_count(obj):

        shape = getattr(
            obj,
            "shape",
            None,
        )

        if shape is not None:

            try:
                return int(shape[0])

            except (
                IndexError,
                TypeError,
                ValueError,
            ):
                return None

        try:
            return len(obj)

        except TypeError:
            return None
