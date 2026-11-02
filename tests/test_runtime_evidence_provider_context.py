from types import SimpleNamespace

from pipediag.evidence.runtime_provider import RuntimeEvidenceProvider


class FakeEvidenceCollector:
    def get_dataset_references(self):
        return {
            "D-001": SimpleNamespace(dataset_id="D-001"),
            "D-002": SimpleNamespace(dataset_id="D-002"),
        }


class FakeRuntimeContext:
    def __init__(self):
        self._events = [
            SimpleNamespace(
                event_id="E-001",
                operation="fit",
                status="success",
            ),
        ]
        self._evaluations = [
            SimpleNamespace(
                evaluation_id="EV-001",
                metric_name="accuracy",
                metric_value=0.92,
                evaluation_context="TRAIN",
            ),
        ]
        self.evidence_collector = FakeEvidenceCollector()

    def get_events(self):
        return list(self._events)

    def get_evaluations(self):
        return list(self._evaluations)


def test_runtime_provider_accepts_run_context():
    context = FakeRuntimeContext()

    provider = RuntimeEvidenceProvider()

    evidence = provider.collect(context)

    assert len(evidence) == 1

    item = evidence[0]

    assert item.evidence_type == "execution_problem"
    assert item.status == "OBSERVED"

    assert item.measurements["value"] == 0.0
    assert item.measurements["execution_failure_rate"] == 0.0


def test_runtime_provider_reads_existing_evaluations():
    context = FakeRuntimeContext()

    provider = RuntimeEvidenceProvider()

    evaluations = provider._get_evaluations(context)

    assert len(evaluations) == 1
    assert evaluations[0].evaluation_id == "EV-001"
    assert evaluations[0].metric_name == "accuracy"
    assert evaluations[0].metric_value == 0.92
    assert evaluations[0].evaluation_context == "TRAIN"




def test_runtime_provider_groups_evaluations_by_context():
    context = FakeRuntimeContext()

    context._evaluations = [
        SimpleNamespace(
            evaluation_id="EV-001",
            metric_name="accuracy",
            metric_value=0.95,
            evaluation_context="TRAIN",
        ),
        SimpleNamespace(
            evaluation_id="EV-002",
            metric_name="accuracy",
            metric_value=0.85,
            evaluation_context="TEST",
        ),
    ]

    provider = RuntimeEvidenceProvider()

    grouped = provider._get_evaluations_by_context(context)

    assert set(grouped) == {"TRAIN", "TEST"}
    assert len(grouped["TRAIN"]) == 1
    assert len(grouped["TEST"]) == 1
    assert grouped["TRAIN"][0].metric_value == 0.95
    assert grouped["TEST"][0].metric_value == 0.85






def test_runtime_provider_builds_generalization_evidence():
    context = FakeRuntimeContext()

    context._evaluations = [
        SimpleNamespace(
            evaluation_id="EV-001",
            metric_name="accuracy",
            metric_value=0.95,
            evaluation_context="TRAIN",
        ),
        SimpleNamespace(
            evaluation_id="EV-002",
            metric_name="accuracy",
            metric_value=0.85,
            evaluation_context="TEST",
        ),
    ]

    provider = RuntimeEvidenceProvider()

    evidence = provider._build_generalization_evidence(context)

    assert evidence.evidence_type == "generalization"
    assert evidence.status == "OBSERVED"
    assert evidence.measurements["gap"] == 0.10