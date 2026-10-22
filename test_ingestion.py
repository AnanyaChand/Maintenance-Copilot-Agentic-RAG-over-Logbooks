from pathlib import Path

import pytest

from maintenance_copilot.ingestion import IngestionError, detect_schema, ingest_csv


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_maintnet.csv"


def test_detects_real_dataset_headers():
    assert detect_schema(["IDENT", "PROBLEM", "ACTION"]) == {
        "record_id": "IDENT",
        "problem": "PROBLEM",
        "action": "ACTION",
    }


def test_ingests_aliases_optional_fields_and_metadata():
    result = ingest_csv(FIXTURE)

    assert result.report.loaded_records == 2
    assert result.report.detected_columns["record_id"] == "Work Order"
    first = result.records[0]
    assert first.record_id == "T-001"
    assert first.domain == "aviation"
    assert first.problem_original == "R/H FWD BAFL SEEAL loose"
    assert first.problem_normalized == "right hand forward baffle seal loose"
    assert first.action_normalized == "removed and replaced seal"
    assert first.ata_chapter == "71"
    assert first.opened_at.isoformat() == "2012-07-02"
    assert first.closed_at.isoformat() == "2012-07-03"
    assert first.metadata == {"Shop": "Powerplant"}


def test_missing_required_column_is_rejected():
    with pytest.raises(IngestionError, match="Missing required canonical fields"):
        detect_schema(["IDENT", "PROBLEM"])


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicates.csv"
    path.write_text(
        "IDENT,PROBLEM,ACTION\n1,first,action one\n1,second,action two\n",
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match="Duplicate record IDs detected: 1"):
        ingest_csv(path)


def test_blank_required_value_is_rejected(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("IDENT,PROBLEM,ACTION\n1,,fixed\n", encoding="utf-8")

    with pytest.raises(IngestionError, match="Blank required field 'problem'"):
        ingest_csv(path)


def test_source_file_is_not_modified():
    before = FIXTURE.read_bytes()
    ingest_csv(FIXTURE)
    assert FIXTURE.read_bytes() == before


def test_original_text_is_preserved_verbatim(tmp_path):
    path = tmp_path / "original.csv"
    path.write_text(
        'IDENT,PROBLEM,ACTION\n1,"  MIXED Case fault.  ","  CHKD ENG.  "\n',
        encoding="utf-8",
    )

    record = ingest_csv(path).records[0]

    assert record.problem_original == "  MIXED Case fault.  "
    assert record.action_original == "  CHKD ENG.  "
    assert record.problem_normalized == "mixed case fault"
    assert record.action_normalized == "checked engine"

