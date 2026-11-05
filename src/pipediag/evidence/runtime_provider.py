from __future__ import annotations

from typing import Any

from pipediag.diagnostics.leakage import LeakageDiagnostic

from pipediag.diagnostics.evidence import (
    DiagnosticEvidence,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_OBSERVED,
)


import numpy as np

from pipediag.diagnostics.distribution_shift import (
    DistributionShiftDiagnostic,
)

from pipediag.diagnostics.imbalance import (
    ImbalanceDiagnostic,
    IMBALANCED,
    NO_IMBALANCE,
)

from pipediag.diagnostics.adapters import (
    distribution_shift_to_evidence,
    imbalance_to_evidence,
)

from pipediag.evidence.models import (
    FEATURE_DATA,
    TARGET_DATA,
)

from pipediag.diagnostics.adapters import (
    distribution_shift_to_evidence,
)

from pipediag.evidence.models import FEATURE_DATA
from pipediag.evidence.store import TRAIN, TEST

from pipediag.diagnostics.generalization import (
    HIGHER_IS_BETTER,
    GeneralizationDiagnostic,
)

SUCCESS_STATUS = "success"


class RuntimeEvidenceProvider:
    """
    Acquire diagnostic evidence directly from a completed PipeDiag
    RunContext.

    V1 exposes:
        - G1: Runtime leakage findings
        - G2: Generalization evidence when evaluations are available
        - G8: Runtime execution evidence

    The provider does not execute the pipeline, inspect source code,
    or perform Bayesian inference.
    """

    def collect(self, source: Any) -> list[DiagnosticEvidence]:
        if source is None:
            raise ValueError("runtime evidence source is required")

        get_events = getattr(source, "get_events", None)

        if not callable(get_events):
            raise TypeError(
                "runtime evidence source must provide get_events()"
            )

        events = get_events()

        if events is None:
            raise ValueError("runtime context returned None for events")

        dataset_references = self._get_dataset_references(source)

        execution_evidence = self._build_execution_evidence(events)

        execution_evidence.metadata["dataset_reference_count"] = len(
            dataset_references
        )

        execution_evidence.metadata["dataset_ids"] = sorted(
            dataset_references
        )

        collected_evidence = [execution_evidence]

        leakage_evidence = self._build_leakage_evidence(
            source=source,
            events=events,
            dataset_references=dataset_references,
        )
        collected_evidence.append(leakage_evidence)



        # Collect generalization evidence when evaluation observations
        # are available in the runtime context.
        if self._get_evaluations(source):
            generalization_evidence = (
                self._build_generalization_evidence(source)
            )
            collected_evidence.append(generalization_evidence)



        evaluation_problem_evidence = (
            self._build_evaluation_problem_evidence(source)
        )
        collected_evidence.append(evaluation_problem_evidence)


        evidence_collector = getattr(
            source,
            "evidence_collector",
            None,
        )

        if (
            evidence_collector is not None
            and getattr(source, "dataset_contexts", None) is not None
        ):
            collected_evidence.append(
                self._build_distribution_shift_evidence(
                    source=source,
                    dataset_references=dataset_references,
                )
            )


            collected_evidence.append(
                self._build_imbalance_evidence(
                    source=source,
                    dataset_references=dataset_references,
                )
            )



            collected_evidence.append(
                self._build_data_quality_evidence(
                    source=source,
                    dataset_references=dataset_references,
                )
            )


            collected_evidence.append(
                self._build_feature_problem_evidence(
                    source=source,
                    dataset_references=dataset_references,
                )
            )


        return collected_evidence



    def _build_leakage_evidence(
        self,
        *,
        source: Any,
        events: list[Any],
        dataset_references: dict,
    ) -> DiagnosticEvidence:
        if not events:
            return DiagnosticEvidence(
                evidence_type="leakage",
                source="runtime",
                status=EVIDENCE_INSUFFICIENT,
                reason="No runtime events were available for leakage analysis.",
                metadata={
                    "inference_eligible": False,
                    "findings": [],
                },
            )

        diagnostic = LeakageDiagnostic(
            context_registry=getattr(
                source,
                "dataset_contexts",
                None,
            )
        )

        findings = diagnostic.analyze(
            events=events,
            dataset_references=dataset_references,
        )

        serialized_findings = [
            {
                "leakage_type": item.leakage_type,
                "status": item.status,
                "rule": item.rule,
                "event_id": item.event_id,
                "operation": item.operation,
                "input_dataset_ids": item.input_dataset_ids,
                "output_dataset_ids": item.output_dataset_ids,
                "violating_dataset_ids": item.violating_dataset_ids,
                "reason": item.reason,
            }
            for item in findings
        ]

        dataset_ids = sorted({
            dataset_id
            for item in findings
            for dataset_id in (
                item.input_dataset_ids
                + item.output_dataset_ids
                + item.violating_dataset_ids
            )
        })



        # Calculate runtime train/test contamination using distinct
        # state-learning operations, not individual diagnostic findings.
        state_learning_operations = {}

        for index, item in enumerate(serialized_findings):
            if (
                item["leakage_type"] != "TRAIN_TEST_CONTAMINATION"
                or item["operation"] not in {"fit", "fit_transform"}
            ):
                continue

            event_id = item["event_id"]
            operation_key = (
                event_id
                if event_id is not None
                else f"missing-event-id-{index}"
            )

            state_learning_operations[operation_key] = item

        operation_statuses = [
            item["status"]
            for item in state_learning_operations.values()
        ]

        confirmed_operation_count = operation_statuses.count(
            "CONFIRMED_LEAKAGE"
        )

        known_clean_operation_count = operation_statuses.count(
            "NO_EVIDENCE"
        )

        unknown_context_count = operation_statuses.count(
            "INSUFFICIENT_EVIDENCE"
        )

        eligible_operation_count = (
            confirmed_operation_count + known_clean_operation_count
        )

        unclassified_operation_count = (
            len(operation_statuses)
            - eligible_operation_count
            - unknown_context_count
        )

        raw_runtime_contamination_rate = (
            confirmed_operation_count / eligible_operation_count
            if eligible_operation_count > 0
            else None
        )

        runtime_contamination_rate = raw_runtime_contamination_rate


        return DiagnosticEvidence(
            evidence_type="leakage",
            source="runtime",
            dataset_ids=dataset_ids,
            status=EVIDENCE_OBSERVED,
            measurements={
                "runtime_contamination_rate": runtime_contamination_rate,
            },            
            metadata={
                # Keep inference disabled until the matching calibrated
                # likelihood model is registered in the next step.
                "inference_eligible": (
                    eligible_operation_count > 0
                    and unknown_context_count == 0
                    and unclassified_operation_count == 0
                    and runtime_contamination_rate is not None
                ),

                "raw_runtime_contamination_rate": raw_runtime_contamination_rate,

                "runtime_contamination_rate": runtime_contamination_rate,
                "state_learning_operation_count": len(
                    state_learning_operations
                ),
                "confirmed_operation_count": confirmed_operation_count,
                "known_clean_operation_count": known_clean_operation_count,
                "eligible_operation_count": eligible_operation_count,
                "unknown_context_count": unknown_context_count,
                "unclassified_operation_count": (
                    unclassified_operation_count
                ),
                
                "finding_count": len(serialized_findings),
                "confirmed_count": sum(
                    item["status"] == "CONFIRMED_LEAKAGE"
                    for item in serialized_findings
                ),
                "potential_count": sum(
                    item["status"] == "POTENTIAL_LEAKAGE"
                    for item in serialized_findings
                ),
                "insufficient_count": sum(
                    item["status"] == "INSUFFICIENT_EVIDENCE"
                    for item in serialized_findings
                ),
                "no_evidence_count": sum(
                    item["status"] == "NO_EVIDENCE"
                    for item in serialized_findings
                ),
                "findings": serialized_findings,
            },
        )



    def _build_execution_evidence(
        self,
        events: list[Any],
    ) -> DiagnosticEvidence:
        total_events = len(events)

        if total_events == 0:
            return DiagnosticEvidence(
                evidence_type="execution_problem",
                source="runtime",
                measurements={},
                status=EVIDENCE_INSUFFICIENT,
                reason=(
                    "No runtime execution events were observed, "
                    "so execution failure rate cannot be determined."
                ),
                metadata={
                    "total_event_count": 0,
                    "failed_event_count": 0,
                    "failed_event_ids": [],
                    "failed_operations": [],
                },
            )

        failed_events = [
            event
            for event in events
            if getattr(event, "status", None) != SUCCESS_STATUS
        ]

        failed_event_ids = [
            event_id
            for event in failed_events
            if (event_id := getattr(event, "event_id", None))
            is not None
        ]

        failed_operations = [
            operation
            for operation in (
                getattr(event, "operation", None)
                for event in failed_events
            )
            if operation is not None
        ]

        failure_rate = len(failed_events) / total_events

        return DiagnosticEvidence(
            evidence_type="execution_problem",
            source="runtime",
            measurements={
                "value": float(failure_rate),
                "execution_failure_rate": float(failure_rate),
                "total_event_count": total_events,
                "failed_event_count": len(failed_events),
            },
            status=EVIDENCE_OBSERVED,
            reason=(
                "Execution failure rate was computed from the "
                "observed runtime events."
            ),
            metadata={
                "metric": "execution_failure_rate",
                "total_event_count": total_events,
                "failed_event_count": len(failed_events),
                "failed_event_ids": failed_event_ids,
                "failed_operations": failed_operations,
            },
        )




    def _get_dataset_references(self, source: Any) -> dict:
        evidence_collector = getattr(
            source,
            "evidence_collector",
            None,
        )

        if evidence_collector is None:
            return {}

        get_dataset_references = getattr(
            evidence_collector,
            "get_dataset_references",
            None,
        )

        if not callable(get_dataset_references):
            return {}

        references = get_dataset_references()

        if references is None:
            return {}

        return dict(references)





    def _get_evaluations(self, source: Any) -> list[Any]:
        get_evaluations = getattr(source, "get_evaluations", None)

        if not callable(get_evaluations):
            return []

        evaluations = get_evaluations()

        if evaluations is None:
            return []

        return list(evaluations)



    def _get_evaluation_count(self, source: Any) -> int:
        return len(self._get_evaluations(source))




    def _get_evaluations_by_context(
        self,
        source: Any,
    ) -> dict[str, list[Any]]:
        evaluations = self._get_evaluations(source)

        grouped: dict[str, list[Any]] = {}

        for evaluation in evaluations:
            context = getattr(
                evaluation,
                "evaluation_context",
                None,
            )

            if context is None:
                continue

            grouped.setdefault(context, []).append(evaluation)

        return grouped



    def _build_generalization_evidence(
        self,
        source: Any,
    ) -> DiagnosticEvidence:
        evaluations = self._get_evaluations(source)

        if not evaluations:
            return DiagnosticEvidence(
                evidence_type="generalization",
                source="runtime",
                measurements={},
                status=EVIDENCE_INSUFFICIENT,
                reason="No evaluation evidence is available.",
            )

        diagnostic = GeneralizationDiagnostic()

        result = diagnostic.analyze(
            evaluations=evaluations,
            metric_direction=HIGHER_IS_BETTER,
        )

        from pipediag.diagnostics.adapters import (
            generalization_to_evidence,
        )

        return generalization_to_evidence(result)


    





    def _build_distribution_shift_evidence(
        self,
        *,
        source: Any,
        dataset_references: dict,
    ) -> DiagnosticEvidence:
        from pipediag.evidence.models import FEATURE_DATA
        from pipediag.evidence.store import TRAIN, TEST

        evidence_collector = getattr(
            source,
            "evidence_collector",
            None,
        )

        registry = getattr(
            evidence_collector,
            "registry",
            None,
        )

        context_registry = getattr(
            source,
            "dataset_contexts",
            None,
        )

        if registry is None or context_registry is None:
            return self._insufficient_shift_evidence(
                "Dataset registry or context registry is unavailable."
            )

        train_candidates = []
        test_candidates = []

        for dataset_id, reference in dataset_references.items():
            if reference.role != FEATURE_DATA:
                continue

            context = context_registry.get_context(dataset_id)

            if context == TRAIN:
                train_candidates.append(reference)
            elif context == TEST:
                test_candidates.append(reference)

        # Find train/test datasets originating from the same source.
        matched_pairs = []

        for train_ref in train_candidates:
            for test_ref in test_candidates:
                common_parents = (
                    set(train_ref.parent_dataset_ids)
                    & set(test_ref.parent_dataset_ids)
                )

                if common_parents:
                    matched_pairs.append(
                        (train_ref, test_ref)
                    )

        if not matched_pairs:
            return self._insufficient_shift_evidence(
                "No related TRAIN and TEST feature datasets were found."
            )

        # Prefer the pair with the largest matching feature count.
        matched_pairs.sort(
            key=lambda pair: (
                min(
                    pair[0].feature_count or 0,
                    pair[1].feature_count or 0,
                ),
                pair[0].dataset_id,
            ),
            reverse=True,
        )

        train_ref, test_ref = matched_pairs[0]


        print("\n========== G3 DEBUG ==========")
        print("TRAIN dataset ID:", train_ref.dataset_id)
        print("TEST dataset ID:", test_ref.dataset_id)
        print("TRAIN feature count:", train_ref.feature_count)
        print("TEST feature count:", test_ref.feature_count)
        print("TRAIN parents:", train_ref.parent_dataset_ids)
        print("TEST parents:", test_ref.parent_dataset_ids)

        train_data = registry.get_object(train_ref.dataset_id)
        test_data = registry.get_object(test_ref.dataset_id)

        if train_data is None or test_data is None:
            return self._insufficient_shift_evidence(
                "The registered TRAIN or TEST dataset object is unavailable.",
                [train_ref.dataset_id, test_ref.dataset_id],
            )


        print("\n========== G3 OBJECT DEBUG ==========")

        for label, dataset in [
            ("TRAIN", train_data),
            ("TEST", test_data),
        ]:
            print(f"\n{label}")
            print("Type:", type(dataset))
            print("Shape:", getattr(dataset, "shape", None))
            print("Has columns:", hasattr(dataset, "columns"))
            print("Has toarray:", hasattr(dataset, "toarray"))

            if hasattr(dataset, "columns"):
                print("Columns:", list(dataset.columns))

            elif hasattr(dataset, "toarray"):
                print("Sparse shape:", dataset.shape)

            else:
                try:
                    import numpy as np
                    array = np.asarray(dataset)
                    print("Array shape:", array.shape)
                    print("Array dtype:", array.dtype)
                except Exception as exc:
                    print("Conversion error:", repr(exc))

        print("====================================")

        train_distributions = self._extract_feature_distributions(
            train_data
        )
        test_distributions = self._extract_feature_distributions(
            test_data
        )


        print("TRAIN extracted features:", len(train_distributions))
        print("TEST extracted features:", len(test_distributions))

        for feature, values in train_distributions.items():
            print(
                "TRAIN",
                feature,
                "count:", len(values),
                "std:", float(np.std(values)),
            )

        if not train_distributions or not test_distributions:
            return self._insufficient_shift_evidence(
                "Numeric feature distributions could not be extracted.",
                [train_ref.dataset_id, test_ref.dataset_id],
            )

        diagnostic = DistributionShiftDiagnostic()

        result = diagnostic.analyze(
            reference_evidence={
                "dataset_id": train_ref.dataset_id,
                "context": TRAIN,
                "distributions": train_distributions,
            },
            comparison_evidence={
                "dataset_id": test_ref.dataset_id,
                "context": TEST,
                "distributions": test_distributions,
            },
        )

        evidence = distribution_shift_to_evidence(result)

        # Calculate the exact measurement used by H3 calibration.
        common_features = (
            set(train_distributions)
            & set(test_distributions)
        )

        standardized_differences = []

        for feature in common_features:
            train_values = np.asarray(
                train_distributions[feature],
                dtype=float,
            )
            test_values = np.asarray(
                test_distributions[feature],
                dtype=float,
            )

            train_std = np.std(train_values)

            if train_std <= 1e-12:
                continue

            difference = abs(
                np.mean(train_values)
                - np.mean(test_values)
            ) / train_std

            standardized_differences.append(
                float(difference)
            )

        if not standardized_differences:
            return self._insufficient_shift_evidence(
                "No features with non-zero training variance were available.",
                [train_ref.dataset_id, test_ref.dataset_id],
            )

        mean_absolute_standardized_shift = float(
            np.mean(standardized_differences)
        )

        evidence.measurements[
            "mean_absolute_standardized_shift"
        ] = mean_absolute_standardized_shift

        evidence.measurements["value"] = (
            mean_absolute_standardized_shift
        )

        evidence.metadata.update({
            "inference_eligible": True,
            "reference_context": TRAIN,
            "comparison_context": TEST,
            "metric": "wasserstein_distance",
            "feature_count": len(result.feature_shifts),
            "diagnostic": "DistributionShiftDiagnostic",
        })

        return evidence

    @staticmethod
    def _extract_feature_distributions(
        dataset: Any,
    ) -> dict[str, list[float]]:
        import numpy as np

        if hasattr(dataset, "toarray"):
            dataset = dataset.toarray()

        # Preserve DataFrame feature names.
        if hasattr(dataset, "columns"):
            distributions = {}

            for column in dataset.columns:
                try:
                    values = dataset[column].to_numpy()
                    numeric = np.asarray(values, dtype=float)

                    if (
                        numeric.ndim == 1
                        and numeric.size > 0
                        and np.isfinite(numeric).all()
                    ):
                        distributions[str(column)] = (
                            numeric.tolist()
                        )

                except (TypeError, ValueError):
                    continue

            return distributions

        try:
            array = np.asarray(dataset, dtype=float)
        except (TypeError, ValueError):
            return {}

        if array.ndim != 2 or array.shape[0] == 0:
            return {}

        distributions = {}

        for index in range(array.shape[1]):
            values = array[:, index]

            if (
                values.size > 0
                and np.isfinite(values).all()
            ):
                distributions[f"feature_{index}"] = (
                    values.tolist()
                )

        return distributions

    @staticmethod
    def _insufficient_shift_evidence(
        reason: str,
        dataset_ids: list[str] | None = None,
    ) -> DiagnosticEvidence:
        return DiagnosticEvidence(
            evidence_type="distribution_shift",
            source="runtime",
            dataset_ids=dataset_ids or [],
            contexts=[],
            measurements={},
            status=EVIDENCE_INSUFFICIENT,
            reason=reason,
            metadata={
                "inference_eligible": False,
                "diagnostic": "DistributionShiftDiagnostic",
            },
        )





    def _build_imbalance_evidence(
        self,
        *,
        source: Any,
        dataset_references: dict,
    ) -> DiagnosticEvidence:

        evidence_collector = getattr(
            source,
            "evidence_collector",
            None,
        )

        registry = getattr(
            evidence_collector,
            "registry",
            None,
        )

        context_registry = getattr(
            source,
            "dataset_contexts",
            None,
        )

        if registry is None or context_registry is None:
            return self._insufficient_imbalance_evidence(
                "Dataset registry or context registry is unavailable."
            )

        train_targets = [
            reference
            for dataset_id, reference in dataset_references.items()
            if (
                reference.role == TARGET_DATA
                and context_registry.get_context(dataset_id) == TRAIN
            )
        ]

        if not train_targets:
            return self._insufficient_imbalance_evidence(
                "No TRAIN target dataset was found."
            )

        train_targets.sort(
            key=lambda reference: reference.dataset_id
        )

        target_ref = train_targets[0]

        target_data = registry.get_object(
            target_ref.dataset_id
        )

        if target_data is None:
            return self._insufficient_imbalance_evidence(
                "Registered TRAIN target data is unavailable.",
                [target_ref.dataset_id],
            )

        try:
            target_array = np.asarray(target_data)

            if target_array.ndim == 2 and target_array.shape[1] == 1:
                target_array = target_array.ravel()

            if target_array.ndim != 1 or target_array.size == 0:
                raise ValueError("Invalid target shape.")

            unique, counts = np.unique(
                target_array,
                return_counts=True,
            )

            total = counts.sum()

            class_distribution = {
                (
                    label.item()
                    if hasattr(label, "item")
                    else label
                ): float(count / total)
                for label, count in zip(unique, counts)
            }

        except (TypeError, ValueError):
            return self._insufficient_imbalance_evidence(
                "TRAIN target class distribution could not be calculated.",
                [target_ref.dataset_id],
            )

        diagnostic = ImbalanceDiagnostic()

        result = diagnostic.analyze(
            target_evidence={
                "dataset_id": target_ref.dataset_id,
                "context": TRAIN,
                "class_distribution": class_distribution,
            }
        )

        evidence = imbalance_to_evidence(result)

        inference_eligible = result.status in {
            IMBALANCED,
            NO_IMBALANCE,
        }

        evidence.measurements["minority_class_proportion"] = (
            result.minority_proportion
        )

        evidence.measurements["value"] = (
            result.minority_proportion
        )

        evidence.metadata.update({
            "inference_eligible": inference_eligible,
            "diagnostic": "ImbalanceDiagnostic",
            "metric": "minority_class_proportion",
            "class_distribution": class_distribution,
        })

        return evidence

    @staticmethod
    def _insufficient_imbalance_evidence(
        reason: str,
        dataset_ids: list[str] | None = None,
    ) -> DiagnosticEvidence:

        return DiagnosticEvidence(
            evidence_type="imbalance",
            source="runtime",
            dataset_ids=dataset_ids or [],
            contexts=[],
            measurements={},
            status=EVIDENCE_INSUFFICIENT,
            reason=reason,
            metadata={
                "inference_eligible": False,
                "diagnostic": "ImbalanceDiagnostic",
            },
        )










    def _build_data_quality_evidence(
        self,
        *,
        source: Any,
        dataset_references: dict,
    ) -> DiagnosticEvidence:

        evidence_collector = getattr(
            source,
            "evidence_collector",
            None,
        )

        registry = getattr(
            evidence_collector,
            "registry",
            None,
        )

        context_registry = getattr(
            source,
            "dataset_contexts",
            None,
        )

        if registry is None or context_registry is None:
            return self._insufficient_data_quality_evidence(
                "Dataset registry or context registry is unavailable."
            )

        train_features = [
            reference
            for dataset_id, reference in dataset_references.items()
            if (
                reference.role == FEATURE_DATA
                and context_registry.get_context(dataset_id) == TRAIN
                and not any(
                    context_registry.get_context(parent_id) == TRAIN
                    for parent_id in reference.parent_dataset_ids
                )
            )
        ]

        if not train_features:
            return self._insufficient_data_quality_evidence(
                "No original TRAIN feature dataset was found."
            )

        train_features.sort(
            key=lambda reference: reference.dataset_id
        )

        feature_ref = train_features[0]

        feature_data = registry.get_object(
            feature_ref.dataset_id
        )

        if feature_data is None:
            return self._insufficient_data_quality_evidence(
                "Registered TRAIN feature data is unavailable.",
                [feature_ref.dataset_id],
            )

        try:
            if hasattr(feature_data, "toarray"):
                feature_data = feature_data.toarray()
            elif hasattr(feature_data, "to_numpy"):
                feature_data = feature_data.to_numpy()

            feature_array = np.asarray(feature_data)

            if feature_array.ndim != 2:
                raise ValueError("Features must be two-dimensional.")

            if feature_array.size == 0:
                raise ValueError("Feature dataset is empty.")

            missing_value_rate = float(
                np.isnan(feature_array).mean()
            )

        except (TypeError, ValueError):
            return self._insufficient_data_quality_evidence(
                "Missing-value rate could not be calculated.",
                [feature_ref.dataset_id],
            )

        return DiagnosticEvidence(
            evidence_type="data_quality",
            source="runtime",
            dataset_ids=[feature_ref.dataset_id],
            contexts=[TRAIN],
            measurements={
                "value": missing_value_rate,
                "missing_value_rate": missing_value_rate,
            },
            status=EVIDENCE_OBSERVED,
            reason=(
                "Missing-value rate was calculated from the original "
                "TRAIN feature dataset before transformation."
            ),
            metadata={
                "inference_eligible": True,
                "diagnostic": "DataQualityDiagnostic",
                "metric": "missing_value_rate",
                "missing_value_rate": missing_value_rate,
                "source": "runtime_training_features",
            },
        )

    @staticmethod
    def _insufficient_data_quality_evidence(
        reason: str,
        dataset_ids: list[str] | None = None,
    ) -> DiagnosticEvidence:

        return DiagnosticEvidence(
            evidence_type="data_quality",
            source="runtime",
            dataset_ids=dataset_ids or [],
            contexts=[],
            measurements={},
            status=EVIDENCE_INSUFFICIENT,
            reason=reason,
            metadata={
                "inference_eligible": False,
                "diagnostic": "DataQualityDiagnostic",
                "metric": "missing_value_rate",
            },
        )



    def _build_feature_problem_evidence(
        self,
        *,
        source: Any,
        dataset_references: dict,
    ) -> DiagnosticEvidence:

        evidence_collector = getattr(
            source,
            "evidence_collector",
            None,
        )

        registry = getattr(
            evidence_collector,
            "registry",
            None,
        )

        context_registry = getattr(
            source,
            "dataset_contexts",
            None,
        )

        if registry is None or context_registry is None:
            return self._insufficient_feature_problem_evidence(
                "Dataset registry or context registry is unavailable."
            )

        train_features = [
            reference
            for dataset_id, reference in dataset_references.items()
            if (
                reference.role == FEATURE_DATA
                and context_registry.get_context(dataset_id) == TRAIN
                and not any(
                    context_registry.get_context(parent_id) == TRAIN
                    for parent_id in reference.parent_dataset_ids
                )
            )
        ]

        if not train_features:
            return self._insufficient_feature_problem_evidence(
                "No original TRAIN feature dataset was found."
            )

        train_features.sort(
            key=lambda reference: reference.dataset_id
        )

        feature_ref = train_features[0]
        feature_data = registry.get_object(feature_ref.dataset_id)

        if feature_data is None:
            return self._insufficient_feature_problem_evidence(
                "Registered TRAIN feature data is unavailable.",
                [feature_ref.dataset_id],
            )

        try:
            if hasattr(feature_data, "toarray"):
                feature_data = feature_data.toarray()
            elif hasattr(feature_data, "to_numpy"):
                feature_data = feature_data.to_numpy()

            feature_array = np.asarray(
                feature_data,
                dtype=float,
            )

            if (
                feature_array.ndim != 2
                or feature_array.shape[0] == 0
                or feature_array.shape[1] == 0
            ):
                raise ValueError(
                    "TRAIN features must be a non-empty 2D array."
                )

            constant_feature_count = sum(
                np.allclose(
                    feature_array[:, index],
                    feature_array[0, index],
                )
                for index in range(feature_array.shape[1])
            )

            n_features = int(feature_array.shape[1])

            constant_feature_ratio = float(
                constant_feature_count / n_features
            )

        except (TypeError, ValueError):
            return self._insufficient_feature_problem_evidence(
                "Constant-feature ratio could not be calculated.",
                [feature_ref.dataset_id],
            )

        return DiagnosticEvidence(
            evidence_type="feature_problem",
            source="runtime",
            dataset_ids=[feature_ref.dataset_id],
            contexts=[TRAIN],
            measurements={
                "value": constant_feature_ratio,
                "constant_feature_ratio": constant_feature_ratio,
                "constant_feature_count": int(
                    constant_feature_count
                ),
                "n_features": n_features,
            },
            status=EVIDENCE_OBSERVED,
            reason=(
                "Constant-feature ratio was calculated from the original "
                "TRAIN feature dataset before transformation."
            ),
            metadata={
                "inference_eligible": True,
                "diagnostic": "FeatureProblemDiagnostic",
                "metric": "constant_feature_ratio",
                "constant_feature_ratio": constant_feature_ratio,
                "constant_feature_count": int(
                    constant_feature_count
                ),
                "n_features": n_features,
                "source": "runtime_training_features",
            },
        )

    @staticmethod
    def _insufficient_feature_problem_evidence(
        reason: str,
        dataset_ids: list[str] | None = None,
    ) -> DiagnosticEvidence:

        return DiagnosticEvidence(
            evidence_type="feature_problem",
            source="runtime",
            dataset_ids=dataset_ids or [],
            contexts=[],
            measurements={},
            status=EVIDENCE_INSUFFICIENT,
            reason=reason,
            metadata={
                "inference_eligible": False,
                "diagnostic": "FeatureProblemDiagnostic",
                "metric": "constant_feature_ratio",
            },
        )

    

    def _build_evaluation_problem_evidence(
        self,
        source: Any,
    ) -> DiagnosticEvidence:
        evaluations = self._get_evaluations_by_context(source)
        test_evaluations = evaluations.get(TEST, [])

        accuracy_evaluations = [
            evaluation
            for evaluation in test_evaluations
            if (
                getattr(evaluation, "metric_name", "").lower()
                == "accuracy"
                and getattr(evaluation, "metric_value", None) is not None
                and getattr(evaluation, "sample_count", None) is not None
            )
        ]

        if not accuracy_evaluations:
            return DiagnosticEvidence(
                evidence_type="evaluation_problem",
                source="runtime",
                status=EVIDENCE_INSUFFICIENT,
                reason=(
                    "No TEST accuracy evaluation with a valid sample "
                    "count was available."
                ),
                metadata={
                    "inference_eligible": False,
                    "metric": "evaluation_standard_error",
                },
            )


        accuracy_evaluations = [
            evaluation
            for evaluation in test_evaluations
            if (
                getattr(evaluation, "metric_name", "").lower()
                == "accuracy"
                and getattr(evaluation, "metric_value", None) is not None
                and getattr(evaluation, "sample_count", None) is not None
            )
        ]

        if not accuracy_evaluations:
            return DiagnosticEvidence(
                evidence_type="evaluation_problem",
                source="runtime",
                status=EVIDENCE_INSUFFICIENT,
                reason=(
                    "No TEST accuracy evaluation with a valid sample "
                    "count was available."
                ),
                metadata={
                    "inference_eligible": False,
                    "metric": "wilson_interval_half_width",
                },
            )

        evaluation = accuracy_evaluations[-1]
        accuracy = float(evaluation.metric_value)
        sample_count = int(evaluation.sample_count)

        if not 0.0 <= accuracy <= 1.0 or sample_count <= 0:
            return DiagnosticEvidence(
                evidence_type="evaluation_problem",
                source="runtime",
                status=EVIDENCE_INSUFFICIENT,
                reason="TEST accuracy or sample count is invalid.",
                metadata={
                    "inference_eligible": False,
                    "metric": "wilson_interval_half_width",
                },
            )

        z = 1.96
        n = float(sample_count)
        denominator = 1.0 + (z * z / n)

        wilson_half_width = float(
            z
            * np.sqrt(
                accuracy * (1.0 - accuracy) / n
                + (z * z) / (4.0 * n * n)
            )
            / denominator
        )

        max_wilson_half_width = 0.05
        evaluation_precision_adequate = (
            wilson_half_width <= max_wilson_half_width
        )

        return DiagnosticEvidence(
            evidence_type="evaluation_problem",
            source="runtime",
            dataset_ids=[
                dataset_id
                for dataset_id in (
                    evaluation.feature_dataset_id,
                    evaluation.target_dataset_id,
                    evaluation.prediction_dataset_id,
                )
                if dataset_id is not None
            ],
            contexts=[TEST],
            measurements={
                "value": wilson_half_width,
                "wilson_interval_half_width": wilson_half_width,
                "accuracy": accuracy,
                "evaluation_sample_count": sample_count,
                "log_wilson_interval_half_width": float(
                    np.log(wilson_half_width)
                ),
            },
            status=EVIDENCE_OBSERVED,
            reason=(
                "Wilson 95% confidence interval half-width was calculated "
                "from TEST accuracy and sample count."
            ),
            metadata={
                "inference_eligible": True,
                "metric": "wilson_interval_half_width",
                "evaluation_sample_count": sample_count,
                "accuracy": accuracy,
                "max_wilson_half_width": max_wilson_half_width,
                "evaluation_precision_adequate": (
                    evaluation_precision_adequate
                ),
            },
        )
