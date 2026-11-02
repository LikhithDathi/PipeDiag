from __future__ import annotations

from typing import Any

from pipediag.diagnostics.evidence import (
    DiagnosticEvidence,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_OBSERVED,
)

from pipediag.diagnostics.generalization import (
    HIGHER_IS_BETTER,
    GeneralizationDiagnostic,
)

SUCCESS_STATUS = "success"


class RuntimeEvidenceProvider:
    """
    Acquire diagnostic evidence directly from a completed PipeDiag
    RunContext.

    V1 currently exposes runtime/execution evidence (G8) from the
    events already collected by the runtime instrumentation layer.

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

        return [execution_evidence]


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
