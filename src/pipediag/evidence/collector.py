from __future__ import annotations

from typing import Any

from .lineage import DatasetLineage
from .models import (
    FEATURE_DATA,
    PREDICTION_DATA,
    TARGET_DATA,
    UNKNOWN,
    DatasetReference,
)
from .registry import DatasetRegistry


class EvidenceCollector:
    """
    Converts supported runtime operations into dataset evidence.

    The collector does not inspect source code or infer feature semantics.
    Dataset roles are assigned from the supported sklearn operation contract.

    Dataset context is resolved conservatively through dataset lineage:
    a derived dataset may inherit TRAIN or TEST context from its ancestors
    when all known ancestor contexts agree. Conflicting contexts remain
    UNKNOWN.
    """

    SUPPORTED_OPERATIONS = {
        "fit",
        "transform",
        "fit_transform",
        "predict",
        "predict_proba",
        "decision_function",
        "score",
    }

    def __init__(
        self,
        lineage: DatasetLineage | None = None,
        registry: DatasetRegistry | None = None,
        context_registry=None,
    ) -> None:

        self.lineage = lineage or DatasetLineage()
        self.registry = registry or DatasetRegistry()

        if context_registry is None:
            from .store import DatasetContextRegistry

            context_registry = DatasetContextRegistry()

        self.context_registry = context_registry

        self.evaluation_acquirer = None


    def get_dataset_references(self) -> dict[str, DatasetReference]:
        """Return the dataset references observed during the run."""
        return dict(self.registry.datasets)


    def observe(
        self,
        operation: str,
        inputs: tuple[Any, ...] | list[Any],
        output: Any = None,
    ) -> dict[str, Any]:

        if operation not in self.SUPPORTED_OPERATIONS:
            return {
                "inputs": [],
                "outputs": [],
            }

        inputs = tuple(inputs)

        if operation == "fit":
            return self._observe_fit(inputs)

        if operation == "transform":
            return self._observe_transform(inputs, output)

        if operation == "fit_transform":
            return self._observe_fit_transform(inputs, output)

        if operation in {
            "predict",
            "predict_proba",
            "decision_function",
        }:
            return self._observe_prediction(inputs, output)

        if operation == "score":
            return self._observe_score(
                inputs,
                output,
            )

        return {
            "inputs": [],
            "outputs": [],
        }

    # ------------------------------------------------------------------
    # Operation handlers
    # ------------------------------------------------------------------

    def _observe_fit(
        self,
        inputs: tuple[Any, ...],
    ) -> dict[str, list[DatasetReference]]:

        references: list[DatasetReference] = []

        if len(inputs) >= 1:
            references.append(
                self._reference_for(
                    inputs[0],
                    role=FEATURE_DATA,
                )
            )

        if len(inputs) >= 2:
            references.append(
                self._reference_for(
                    inputs[1],
                    role=TARGET_DATA,
                )
            )

        return {
            "inputs": references,
            "outputs": [],
        }

    def _observe_transform(
        self,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> dict[str, list[DatasetReference]]:

        input_refs: list[DatasetReference] = []

        if len(inputs) >= 1:
            input_refs.append(
                self._reference_for(
                    inputs[0],
                    role=FEATURE_DATA,
                )
            )

        output_refs: list[DatasetReference] = []

        if output is not None:
            output_refs.append(
                self._reference_for(
                    output,
                    role=FEATURE_DATA,
                    parent_dataset_ids=[
                        reference.dataset_id
                        for reference in input_refs
                    ],
                )
            )

        return {
            "inputs": input_refs,
            "outputs": output_refs,
        }

    def _observe_fit_transform(
        self,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> dict[str, list[DatasetReference]]:

        input_refs: list[DatasetReference] = []

        if len(inputs) >= 1:
            input_refs.append(
                self._reference_for(
                    inputs[0],
                    role=FEATURE_DATA,
                )
            )

        if len(inputs) >= 2:
            input_refs.append(
                self._reference_for(
                    inputs[1],
                    role=TARGET_DATA,
                )
            )

        output_refs: list[DatasetReference] = []

        if output is not None:
            feature_inputs = [
                reference
                for reference in input_refs
                if reference.role == FEATURE_DATA
            ]

            output_refs.append(
                self._reference_for(
                    output,
                    role=FEATURE_DATA,
                    parent_dataset_ids=[
                        reference.dataset_id
                        for reference in feature_inputs
                    ],
                )
            )

        return {
            "inputs": input_refs,
            "outputs": output_refs,
        }

    def _observe_prediction(
        self,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> dict[str, list[DatasetReference]]:

        input_refs: list[DatasetReference] = []

        if len(inputs) >= 1:
            input_refs.append(
                self._reference_for(
                    inputs[0],
                    role=FEATURE_DATA,
                )
            )

        output_refs: list[DatasetReference] = []

        if output is not None:
            output_refs.append(
                self._reference_for(
                    output,
                    role=PREDICTION_DATA,
                    parent_dataset_ids=[
                        reference.dataset_id
                        for reference in input_refs
                    ],
                )
            )

        return {
            "inputs": input_refs,
            "outputs": output_refs,
        }

    def _observe_score(
        self,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> dict[str, Any]:

        from pipediag.acquisition.evaluation import EvaluationAcquirer

        input_refs: list[DatasetReference] = []

        if len(inputs) >= 1:
            input_refs.append(
                self._reference_for(
                    inputs[0],
                    role=FEATURE_DATA,
                )
            )

        if len(inputs) >= 2:
            input_refs.append(
                self._reference_for(
                    inputs[1],
                    role=TARGET_DATA,
                )
            )

        # --------------------------------------------------------------
        # Resolve context through dataset lineage before acquiring
        # evaluation evidence.
        #
        # The actual object passed to estimator.score() may be a
        # transformed descendant of the original TRAIN/TEST dataset.
        # Therefore its own context can be UNKNOWN even though its
        # lineage contains a dataset with an established context.
        # --------------------------------------------------------------

        for reference in input_refs:
            self._propagate_lineage_context(reference.dataset_id)

        feature_dataset_id = None
        target_dataset_id = None

        for reference in input_refs:

            if reference.role == FEATURE_DATA:
                feature_dataset_id = reference.dataset_id

            elif reference.role == TARGET_DATA:
                target_dataset_id = reference.dataset_id

        if self.evaluation_acquirer is None:
            self.evaluation_acquirer = EvaluationAcquirer(
                context_registry=self.context_registry
            )

        evaluation = self.evaluation_acquirer.acquire_score(
            metric_name="score",
            metric_value=output,
            feature_dataset_id=feature_dataset_id,
            target_dataset_id=target_dataset_id,
        )

        return {
            "inputs": input_refs,
            "outputs": [],
            "evaluation": evaluation,
        }

    # ------------------------------------------------------------------
    # Dataset context propagation
    # ------------------------------------------------------------------

    def _propagate_lineage_context(
        self,
        dataset_id: str,
    ) -> str:
        """
        Resolve the context of a dataset from its lineage.

        Rules:

        1. An explicitly established context on the dataset itself wins.
        2. If the dataset has no known context, inspect its ancestors.
        3. If all known ancestor contexts agree, inherit that context.
        4. If known ancestors contain conflicting contexts, use UNKNOWN.
        5. If no ancestor has a known context, use UNKNOWN.

        This method deliberately does not infer context from operation names
        such as fit, predict, or score.
        """

        existing_context = self.context_registry.get_context(
            dataset_id
        )

        if existing_context != UNKNOWN:
            return existing_context

        reference = self.lineage.get(dataset_id)

        if reference is None:
            return UNKNOWN

        ancestor_contexts = self._ancestor_contexts(
            reference.parent_dataset_ids
        )

        known_contexts = {
            context
            for context in ancestor_contexts
            if context != UNKNOWN
        }

        if len(known_contexts) == 1:
            resolved_context = next(iter(known_contexts))

            self.context_registry.set_context(
                dataset_id,
                resolved_context,
            )

            return resolved_context

        # Either there was no known context or there were conflicting
        # contexts. Both cases must remain UNKNOWN.
        if len(known_contexts) > 1:
            self.context_registry.set_context(
                dataset_id,
                UNKNOWN,
            )

        return UNKNOWN

    def _ancestor_contexts(
        self,
        parent_dataset_ids: list[str],
    ) -> list[str]:
        """
        Recursively collect contexts from all ancestors.

        The visited set prevents cycles from causing infinite recursion.
        """

        contexts: list[str] = []
        visited: set[str] = set()

        def visit(dataset_id: str) -> None:

            if dataset_id in visited:
                return

            visited.add(dataset_id)

            context = self.context_registry.get_context(
                dataset_id
            )

            contexts.append(context)

            reference = self.lineage.get(dataset_id)

            if reference is None:
                return

            for parent_id in reference.parent_dataset_ids:
                visit(parent_id)

        for parent_id in parent_dataset_ids:
            visit(parent_id)

        return contexts

    # ------------------------------------------------------------------
    # Dataset reference management
    # ------------------------------------------------------------------

    def _reference_for(
        self,
        obj: Any,
        role: str,
        parent_dataset_ids: list[str] | None = None,
    ) -> DatasetReference:

        existing = self.registry.get(obj)

        if existing is not None:

            # Do not overwrite an already established role.
            # This prevents nested calls from changing the identity
            # of an existing dataset reference.
            if existing.role == UNKNOWN:
                existing.role = role

            return existing

        shape = self._extract_shape(obj)
        dtype = self._extract_dtype(obj)

        reference = self.lineage.create_reference(
            shape=shape,
            dtype=dtype,
            role=role,
            object_id=id(obj),
            parent_dataset_ids=parent_dataset_ids or [],
        )

        self.registry.register(
            obj,
            reference,
        )

        return reference

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_shape(
        obj: Any,
    ) -> tuple[int, ...] | None:

        shape = getattr(obj, "shape", None)

        if shape is None:
            return None

        try:
            return tuple(int(value) for value in shape)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_dtype(
        obj: Any,
    ) -> str | None:

        dtype = getattr(obj, "dtype", None)

        if dtype is None:
            return None

        return str(dtype)