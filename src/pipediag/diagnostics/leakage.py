
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipediag.evidence.models import (
    FEATURE_DATA,
    TARGET_DATA,
    DatasetReference,
)

from pipediag.evidence.store import (
    DatasetContextRegistry,
    TRAIN,
    TEST,
    VALIDATION,
    UNKNOWN,
)


# ======================================================================
# Leakage types
# ======================================================================

TRAIN_TEST_CONTAMINATION = "TRAIN_TEST_CONTAMINATION"
TARGET_FEATURE_DEPENDENCY = "TARGET_FEATURE_DEPENDENCY"


# ======================================================================
# Leakage statuses
# ======================================================================

NO_EVIDENCE = "NO_EVIDENCE"
POTENTIAL_LEAKAGE = "POTENTIAL_LEAKAGE"
CONFIRMED_LEAKAGE = "CONFIRMED_LEAKAGE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ======================================================================
# LeakageEvidence
# ======================================================================


@dataclass
class LeakageEvidence:
    """
    Derived evidence describing an observed or potentially
    problematic information flow within an ML pipeline.

    This class records the evidence behind a leakage finding.
    It does not determine whether the entire model is invalid.
    """

    evidence_type: str = "leakage"
    leakage_type: str = ""
    status: str = INSUFFICIENT_EVIDENCE
    rule: str = ""
    event_id: str | None = None
    operation: str | None = None

    input_dataset_ids: list[str] = field(
        default_factory=list
    )

    output_dataset_ids: list[str] = field(
        default_factory=list
    )

    violating_dataset_ids: list[str] = field(
        default_factory=list
    )

    reason: str = ""


# ======================================================================
# Leakage Diagnostic
# ======================================================================


class LeakageDiagnostic:
    """
    Detects observable leakage patterns from runtime events,
    dataset roles, and dataset contexts.

    V1 supports:

        1. Train/test contamination
        2. Target-to-feature dependency

    The diagnostic does not inspect source code and does not
    attempt to infer hidden data dependencies that are not
    represented in the available evidence.
    """

    STATE_LEARNING_OPERATIONS = {
        "fit",
        "fit_transform",
    }

    def __init__(
        self,
        context_registry: DatasetContextRegistry | None = None,
    ) -> None:

        self.context_registry = (
            context_registry
            or DatasetContextRegistry()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        events: list[Any],
        dataset_references: dict[str, DatasetReference],
    ) -> list[LeakageEvidence]:

        results: list[LeakageEvidence] = []

        for event in events:
            results.append(
                self._analyze_train_test_contamination(
                    event=event,
                    dataset_references=dataset_references,
                )
            )

            results.append(
                self._analyze_target_feature_dependency(
                    event=event,
                    dataset_references=dataset_references,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Train / test contamination
    # ------------------------------------------------------------------

    def _analyze_train_test_contamination(
        self,
        *,
        event: Any,
        dataset_references: dict[str, DatasetReference],
    ) -> LeakageEvidence:

        input_ids = list(
            getattr(event, "input_dataset_refs", [])
        )

        output_ids = list(
            getattr(event, "output_dataset_refs", [])
        )

        event_id = getattr(event, "event_id", None)
        operation = getattr(event, "operation", None)

        base = LeakageEvidence(
            leakage_type=TRAIN_TEST_CONTAMINATION,
            event_id=event_id,
            operation=operation,
            input_dataset_ids=input_ids,
            output_dataset_ids=output_ids,
            rule=(
                "TEST data must not participate in "
                "state-learning operations such as fit "
                "or fit_transform."
            ),
        )

        if operation not in self.STATE_LEARNING_OPERATIONS:
            base.status = NO_EVIDENCE
            base.reason = (
                "The operation does not learn model or "
                "transformer state, so TEST data participation "
                "does not constitute train/test contamination."
            )
            return base

        relevant_ids = [
            dataset_id
            for dataset_id in input_ids
            if self._is_feature_dataset(
                dataset_id,
                dataset_references,
            )
        ]

        if not relevant_ids:
            base.status = INSUFFICIENT_EVIDENCE
            base.reason = (
                "No feature dataset involved in the "
                "state-learning operation could be identified."
            )
            return base

        contexts = {
            dataset_id: self.context_registry.get_context(
                dataset_id
            )
            for dataset_id in relevant_ids
        }

        unknown_ids = [
            dataset_id
            for dataset_id, context in contexts.items()
            if context == UNKNOWN
        ]

        if unknown_ids:
            base.status = INSUFFICIENT_EVIDENCE
            base.violating_dataset_ids = []
            base.reason = (
                "The context of one or more feature datasets "
                "participating in the state-learning operation "
                "is UNKNOWN, so train/test contamination cannot "
                "be determined reliably."
            )
            return base

        test_ids = [
            dataset_id
            for dataset_id, context in contexts.items()
            if context == TEST
        ]

        if test_ids:
            base.status = CONFIRMED_LEAKAGE
            base.violating_dataset_ids = test_ids
            base.reason = (
                "TEST data participated in a state-learning "
                "operation. Information from TEST data could "
                "therefore influence learned state."
            )
            return base

        base.status = NO_EVIDENCE
        base.reason = (
            "No TEST feature dataset participated in the "
            "state-learning operation."
        )

        return base

    # ------------------------------------------------------------------
    # Target -> feature dependency
    # ------------------------------------------------------------------

    def _analyze_target_feature_dependency(
        self,
        *,
        event: Any,
        dataset_references: dict[str, DatasetReference],
    ) -> LeakageEvidence:

        input_ids = list(
            getattr(event, "input_dataset_refs", [])
        )

        output_ids = list(
            getattr(event, "output_dataset_refs", [])
        )

        event_id = getattr(event, "event_id", None)
        operation = getattr(event, "operation", None)

        base = LeakageEvidence(
            leakage_type=TARGET_FEATURE_DEPENDENCY,
            event_id=event_id,
            operation=operation,
            input_dataset_ids=input_ids,
            output_dataset_ids=output_ids,
            rule=(
                "A feature dataset must not be derived from "
                "target data unless the dependency is known "
                "to be legitimate."
            ),
        )

        # For sklearn transform and fit_transform, X is the
        # first positional argument. The optional second
        # argument y is not automatically a feature dependency.
        if operation in {"transform", "fit_transform"}:
            candidate_inputs = input_ids[:1]
        else:
            candidate_inputs = input_ids

        target_inputs = {
            dataset_id
            for dataset_id in candidate_inputs
            if self._has_role(
                dataset_id,
                TARGET_DATA,
                dataset_references,
            )
        }

        feature_outputs = [
            dataset_id
            for dataset_id in output_ids
            if self._has_role(
                dataset_id,
                FEATURE_DATA,
                dataset_references,
            )
        ]

        if not target_inputs or not feature_outputs:
            base.status = NO_EVIDENCE
            base.reason = (
                "No target-to-feature dependency was observed "
                "in this operation."
            )
            return base

        # Inspect recorded lineage where available.
        derived_from_target = []

        for output_id in feature_outputs:
            output_reference = dataset_references.get(output_id)

            if output_reference is None:
                continue

            parents = set(
                output_reference.parent_dataset_ids
            )

            if target_inputs.intersection(parents):
                derived_from_target.append(output_id)

        has_lineage_information = any(
            dataset_references.get(output_id) is not None
            and dataset_references[output_id].parent_dataset_ids
            for output_id in feature_outputs
        )

        if has_lineage_information and not derived_from_target:
            base.status = NO_EVIDENCE
            base.reason = (
                "Target data was passed to the operation, but "
                "the recorded feature-output lineage does not "
                "show a direct target dependency."
            )
            return base

        base.status = POTENTIAL_LEAKAGE
        base.violating_dataset_ids = sorted(target_inputs)

        base.reason = (
            "A target dataset was identified as a possible "
            "source of a feature output. This may introduce "
            "target leakage, but the available evidence does "
            "not establish whether the dependency was legitimate."
        )

        return base

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_role(
        dataset_id: str,
        role: str,
        dataset_references: dict[str, DatasetReference],
    ) -> bool:

        reference = dataset_references.get(dataset_id)

        if reference is None:
            return False

        return reference.role == role

    @staticmethod
    def _is_feature_dataset(
        dataset_id: str,
        dataset_references: dict[str, DatasetReference],
    ) -> bool:

        return LeakageDiagnostic._has_role(
            dataset_id=dataset_id,
            role=FEATURE_DATA,
            dataset_references=dataset_references,
        )
