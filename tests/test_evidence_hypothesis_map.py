from pipediag.diagnostics.evidence_hypothesis_map import (
    EVIDENCE_HYPOTHESIS_MAP,
)


def test_all_evidence_groups_are_defined():
    assert set(EVIDENCE_HYPOTHESIS_MAP) == {
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "G8",
    }


def test_primary_evidence_hypothesis_relationships():
    assert EVIDENCE_HYPOTHESIS_MAP["G1"]["primary"] == {"H1"}
    assert EVIDENCE_HYPOTHESIS_MAP["G2"]["primary"] == {"H2", "H7"}
    assert EVIDENCE_HYPOTHESIS_MAP["G3"]["primary"] == {"H3"}
    assert EVIDENCE_HYPOTHESIS_MAP["G4"]["primary"] == {"H4"}
    assert EVIDENCE_HYPOTHESIS_MAP["G5"]["primary"] == {"H5"}
    assert EVIDENCE_HYPOTHESIS_MAP["G6"]["primary"] == {"H6"}
    assert EVIDENCE_HYPOTHESIS_MAP["G7"]["primary"] == {"H8"}
    assert EVIDENCE_HYPOTHESIS_MAP["G8"]["primary"] == {"H9"}


def test_secondary_evidence_hypothesis_relationships():
    assert EVIDENCE_HYPOTHESIS_MAP["G1"]["secondary"] == set()
    assert EVIDENCE_HYPOTHESIS_MAP["G2"]["secondary"] == set()
    assert EVIDENCE_HYPOTHESIS_MAP["G3"]["secondary"] == set()
    assert EVIDENCE_HYPOTHESIS_MAP["G4"]["secondary"] == set()
    assert EVIDENCE_HYPOTHESIS_MAP["G5"]["secondary"] == set()
    assert EVIDENCE_HYPOTHESIS_MAP["G6"]["secondary"] == set()
    assert EVIDENCE_HYPOTHESIS_MAP["G7"]["secondary"] == set()
    assert EVIDENCE_HYPOTHESIS_MAP["G8"]["secondary"] == set()


def test_no_unknown_hypothesis_ids_are_referenced():
    valid_hypotheses = {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
    }

    for relationships in EVIDENCE_HYPOTHESIS_MAP.values():
        referenced = (
            relationships["primary"]
            | relationships["secondary"]
        )

        assert referenced <= valid_hypotheses


def test_primary_and_secondary_relationships_do_not_overlap():
    for relationships in EVIDENCE_HYPOTHESIS_MAP.values():
        assert (
            relationships["primary"]
            & relationships["secondary"]
        ) == set()


def test_each_evidence_group_has_primary_and_secondary_relationships():
    for relationships in EVIDENCE_HYPOTHESIS_MAP.values():
        assert "primary" in relationships
        assert "secondary" in relationships