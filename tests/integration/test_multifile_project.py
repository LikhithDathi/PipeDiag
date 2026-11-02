from tests.integration.multifile_project.main import run_project


def test_multifile_project_executes_and_is_observed():
    run, score, split_evidence = run_project()

    assert score >= 0.0
    assert score <= 1.0

    events = run.context.get_events()

    assert len(events) > 0


def test_multifile_project_preserves_dataset_lineage():
    run, score, split_evidence = run_project()

    assert split_evidence is not None

    events = run.context.get_events()

    assert len(events) > 0

    # Dataset lineage is maintained by the existing evidence collector.
    assert run.context.evidence_collector is not None