import pytest

from pipediag.application import DiagnosticRunResult
from pipediag.diagnostics.diagnostic_result import DiagnosticResult
from pipediag.diagnostics.explanation import ExplanationSpec
from pipediag.persistence import (
    delete_run,
    list_runs,
    load_run,
    save_run,
)


def make_result():
    diagnostic = DiagnosticResult(
        posterior={"H1": 1.0},
        uncertainty=0.0,
        evidence=[],
        metadata={},
    )

    explanation = ExplanationSpec(
        threshold=0.20,
        uncertainty=0.0,
        hypotheses=(),
        overall_assessment="No dominant explanation identified.",
    )

    return DiagnosticRunResult(
        diagnostic_result=diagnostic,
        explanation=explanation,
        report="Test diagnostic report",
    )


def test_save_and_load_run(tmp_path):
    run_id = save_run(
        make_result(),
        storage_dir=tmp_path,
    )

    record = load_run(
        run_id,
        storage_dir=tmp_path,
    )

    assert record is not None
    assert record["run_id"] == run_id
    assert record["diagnostic"]["posterior"] == {"H1": 1.0}
    assert record["report"] == "Test diagnostic report"


def test_save_run_generates_id(tmp_path):
    run_id = save_run(
        make_result(),
        storage_dir=tmp_path,
    )

    assert run_id.startswith("R-")
    assert load_run(run_id, storage_dir=tmp_path) is not None


def test_list_runs(tmp_path):
    first = save_run(make_result(), storage_dir=tmp_path)
    second = save_run(make_result(), storage_dir=tmp_path)

    run_ids = {
        item["run_id"]
        for item in list_runs(storage_dir=tmp_path)
    }

    assert run_ids == {first, second}


def test_delete_run(tmp_path):
    run_id = save_run(make_result(), storage_dir=tmp_path)

    assert delete_run(run_id, storage_dir=tmp_path) is True
    assert load_run(run_id, storage_dir=tmp_path) is None
    assert delete_run(run_id, storage_dir=tmp_path) is False


def test_load_unknown_run(tmp_path):
    assert load_run(
        "R-unknown",
        storage_dir=tmp_path,
    ) is None






@pytest.mark.parametrize(
    "run_id",
    [
        "../outside",
        "..\\outside",
        "R-../../outside",
        "R-test.json",
        "R test",
        "",
        " ",
        "/absolute/path",
    ],
)
def test_load_run_rejects_invalid_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError):
        load_run(run_id, storage_dir=tmp_path)


@pytest.mark.parametrize(
    "run_id",
    [
        "../outside",
        "..\\outside",
        "R-../../outside",
        "R-test.json",
        "R test",
        "",
        " ",
        "/absolute/path",
    ],
)
def test_delete_run_rejects_invalid_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError):
        delete_run(run_id, storage_dir=tmp_path)


def test_save_run_rejects_invalid_metadata_run_id(tmp_path):
    result = make_result()
    result.diagnostic_result.metadata["run_id"] = "../outside"

    with pytest.raises(ValueError):
        save_run(result, storage_dir=tmp_path)