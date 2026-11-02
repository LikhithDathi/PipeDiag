from datetime import datetime, timezone
from uuid import uuid4

from .events import Event

from pipediag.evidence import EvidenceCollector

from pipediag.evidence.store import DatasetContextRegistry



class RunContext:
    """
    Represents one diagnostic execution.
    """

    def __init__(self) -> None:
        self.run_id = f"R-{uuid4().hex[:8]}"
        self.start_time = datetime.now(timezone.utc)

        self.events: list[Event] = []
        self.evaluations = []

        self._event_counter = 0
        self._event_stack: list[str] = []

        self.dataset_contexts = (
                    DatasetContextRegistry()
                )

        self.evidence_collector = EvidenceCollector(
            context_registry=self.dataset_contexts,
        )

    def next_event_id(self) -> str:
        self._event_counter += 1
        return f"E-{self._event_counter:04d}"

    def begin_event(
        self,
        operation: str,
        component: str | None = None,
        step_name: str | None = None,
        metadata: dict | None = None,
    ) -> Event:

        event_id = self.next_event_id()

        parent_event_id = (
            self._event_stack[-1]
            if self._event_stack
            else None
        )

        depth = len(self._event_stack)

        event = Event(
            event_id=event_id,
            run_id=self.run_id,
            operation=operation,
            component=component,
            step_name=step_name,
            parent_event_id=parent_event_id,
            depth=depth,
            metadata=metadata or {},
        )

        # Store immediately so event order represents execution-start order.
        self.events.append(event)

        self._event_stack.append(event_id)

        return event

    def end_event(
        self,
        event: Event,
        duration_ms: float,
        status: str,
        metadata: dict | None = None,
    ) -> None:

        event.duration_ms = duration_ms
        event.status = status

        if metadata:
            event.metadata.update(metadata)

        if self._event_stack and self._event_stack[-1] == event.event_id:
            self._event_stack.pop()

    def add_event(self, event: Event) -> None:
        """
        Retained for compatibility with the initial event model.
        New instrumentation should use begin_event/end_event.
        """
        self.events.append(event)

    def get_events(self) -> list[Event]:
        return list(self.events)

    def event_count(self) -> int:
        return len(self.events)



    def set_dataset_context(
        self,
        dataset_id: str,
        context: str,
    ) -> None:
        self.dataset_contexts.set_context(
            dataset_id,
            context,
        )


    def get_dataset_context(
        self,
        dataset_id: str,
    ) -> str:
        return self.dataset_contexts.get_context(
            dataset_id,
        )



    def add_evaluation(self, evaluation) -> None:
        if evaluation is None:
            raise ValueError("evaluation is required")

        self.evaluations.append(evaluation)


    def get_evaluations(self) -> list:
        return list(self.evaluations)
