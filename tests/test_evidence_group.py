from pipediag.diagnostics.evidence_group import EvidenceGroup


def test_evidence_group_stores_name_value_and_status():
    evidence = EvidenceGroup(
        name="generalization",
        value=0.31,
        status="observed",
    )

    assert evidence.name == "generalization"
    assert evidence.value == 0.31
    assert evidence.status == "observed"


def test_evidence_group_can_represent_unavailable_evidence():
    evidence = EvidenceGroup(
        name="distribution_shift",
        value=None,
        status="unavailable",
    )

    assert evidence.name == "distribution_shift"
    assert evidence.value is None
    assert evidence.status == "unavailable"


def test_evidence_group_can_store_metadata():
    evidence = EvidenceGroup(
        name="generalization",
        value=0.31,
        status="observed",
        metadata={"train_metric": 0.95, "test_metric": 0.64},
    )

    assert evidence.metadata["train_metric"] == 0.95
    assert evidence.metadata["test_metric"] == 0.64


def test_evidence_group_metadata_defaults_to_empty_dictionary():
    evidence = EvidenceGroup(
        name="generalization",
        value=0.31,
        status="observed",
    )

    assert evidence.metadata == {}


def test_evidence_group_is_immutable():
    evidence = EvidenceGroup(
        name="generalization",
        value=0.31,
        status="observed",
    )

    try:
        evidence.name = "changed"
    except AttributeError:
        pass
    else:
        raise AssertionError("EvidenceGroup should be immutable")


def test_evidence_group_requires_a_name():
    try:
        EvidenceGroup(
            name="",
            value=0.31,
            status="observed",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("EvidenceGroup should require a non-empty name")


def test_evidence_group_requires_a_status():
    try:
        EvidenceGroup(
            name="generalization",
            value=0.31,
            status="",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("EvidenceGroup should require a non-empty status")