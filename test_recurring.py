from types import SimpleNamespace

from maintenance_copilot.ingestion import ingest_csv
from maintenance_copilot.recurring import detect_recurring_patterns


def test_corpus_repeat_requires_cited_match_and_never_runs_on_refusal(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    csv_path.write_text(
        "IDENT,PROBLEM,ACTION\n"
        "1,ENGINE IDLE OVERRIDE KILLED ENGINE.,CHECKED IDLE\n"
        "2,Engine idle override killed engine,REPLACED COMPONENT\n"
        "3,UNRELATED ISSUE,INSPECTED\n",
        encoding="utf-8",
    )
    records = ingest_csv(csv_path).records
    path = tmp_path / "records.jsonl"
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")

    supported = SimpleNamespace(status="answered", cited_record_ids=["1"])
    patterns = detect_recurring_patterns(supported, path)
    assert len(patterns) == 1
    assert patterns[0].record_count == 2
    assert patterns[0].record_ids == ["1", "2"]
    assert detect_recurring_patterns(SimpleNamespace(status="answered", cited_record_ids=["3"]), path) == []
    assert detect_recurring_patterns(SimpleNamespace(status="insufficient_evidence", cited_record_ids=["1"]), path) == []
