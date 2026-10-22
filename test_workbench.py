import csv
import io
import json
from pathlib import Path

import pytest

from maintenance_copilot.workbench import (CaseReview, Catalog, RecordReview, ReviewConflict, ReviewStore,
                                         filter_cases, focused_pack, file_hash)
from maintenance_copilot.workbench_reports import agreement_stats, agreement_report, human_metrics, report


@pytest.fixture(scope="module")
def catalog():
    return Catalog()


@pytest.fixture
def store(tmp_path, catalog):
    return ReviewStore(tmp_path / "reviews", catalog)


def example(catalog, reviewer="Alice"):
    cid = "S01"
    rid = catalog.cases[cid]["answer"]["cited_record_ids"][0]
    return CaseReview(case_id=cid, reviewer_name=reviewer, records={rid: RecordReview(relevance="relevant")})


def test_autosave_reload_snapshots_and_original_unchanged(store, catalog):
    before = file_hash(catalog.original_csv)
    review = example(catalog)
    assert store.save(review, 0) == 1
    loaded, revision = store.load("Alice", "S01")
    assert revision == 1 and loaded.records == review.records
    assert loaded.reviewed_at is not None
    loaded.reviewer_notes = "Independent review"
    assert store.save(loaded, 1) == 2
    assert len(list((store.directory / "snapshots").glob("*.csv"))) == 2
    assert len(store.audit("Alice")) == 2
    assert file_hash(catalog.original_csv) == before
    assert (store.directory / "human_review_v1.csv").exists()


def test_csv_roundtrip_incomplete_and_spreadsheet_escape(store, catalog, tmp_path):
    review = example(catalog)
    review.reviewer_notes = "=do not execute"
    store.save(review, 0)
    text = store.export_csv()
    assert "'=do not execute" in text
    other = ReviewStore(tmp_path / "other", catalog)
    other.import_csv(text, "Importer")
    saved, _ = other.load("Alice", "S01")
    assert saved.reviewer_notes == review.reviewer_notes
    assert saved.answer_grounding is None
    assert not catalog.complete(saved)
    assert not other.import_csv(other.export_csv(), "Importer")
    assert other.audit()[0]["actor"] == "Importer"


def test_conflicts_and_multi_reviewer_preservation(store, catalog):
    review = example(catalog)
    store.save(review, 0)
    with pytest.raises(ReviewConflict):
        store.save(review, 0)
    assert store.load("Alice", "S01")[1] == 1
    assert store.audit()[-1]["kind"] == "revision_conflict"
    bob = example(catalog, "Bob")
    next(iter(bob.records.values())).relevance = "irrelevant"
    store.save(bob, 0)
    result = agreement_report([r for r, _ in store.all()])
    assert result["disagreement_queue"][0]["field"] == "relevance"
    assert result["pairwise"][0]["collapsed_relevance"]["raw_agreement"] == 0
    assert len(store.all()) == 2


def test_bad_ids_and_import_are_atomic(store, catalog):
    review = example(catalog)
    review.suggested_expected_record_ids = ["99999999"]
    with pytest.raises(ValueError, match="Invalid suggested"):
        store.save(review, 0)
    review.suggested_expected_record_ids = []
    review.records["99999999"] = RecordReview()
    with pytest.raises(ValueError, match="Invalid record"):
        store.save(review, 0)
    assert not store.all()
    good = example(catalog)
    store.save(good, 0)
    original = store.export_csv()
    corrupted = original.replace(catalog.fingerprint, "wrong-data")
    with pytest.raises(ValueError, match="Wrong dataset"):
        store.import_csv(corrupted, "Importer")
    assert store.load("Alice", "S01")[1] == 1
    rows = list(csv.DictReader(io.StringIO(original)))
    rows.append(dict(rows[-1]))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ReviewConflict, match="Duplicate record"):
        store.import_csv(output.getvalue(), "Importer")


def test_queue_filters_and_unique_balanced_pack(catalog):
    assert not filter_cases(catalog, {}, search="not a real query")
    assert "S01" in filter_cases(catalog, {}, search="100001")
    assert all(catalog.cases[c]["split"] == "held_out" for c in filter_cases(catalog, {}, split="held_out"))
    assert all(catalog.cases[c]["status"] == "insufficient_evidence" for c in filter_cases(catalog, {}, status="insufficient_evidence"))
    assert all(catalog.cases[c]["suspected_false_confidence"] for c in filter_cases(catalog, {}, flags=["suspected_false_confidence"]))
    assert len(filter_cases(catalog, {}, flags=["action_divergence"])) == 7
    pack = focused_pack(catalog)
    assert pack == focused_pack(catalog)
    assert pack["composition"]["action_divergence"]["selected"] == 7
    assert all(v["selected"] == 10 for k, v in pack["composition"].items() if k != "action_divergence")
    assert len(pack["cases"]) == 57


def test_human_metrics_never_use_provisional_labels(store, catalog):
    assert report(catalog, store)["human_reviewed_labels"] is None
    review = example(catalog)
    store.save(review, 0)
    m = report(catalog, store)
    human = m["human_reviewed_labels"]["Alice"]["all_reviewed"]
    assert human["citation_relevance"]["value"] == 1.
    assert human["answer_grounding_precision"]["value"] is None
    assert human["citation_coverage"]["value"] is None
    assert human["refusal_recall"]["value"] is None
    assert m["provisional_automated_labels"]["held_out"] == catalog.report["held_out"]["metrics"]


def test_human_false_confidence_requires_sufficiency_not_provisional_expectation(catalog):
    review = CaseReview(case_id="M01", reviewer_name="Human", evidence_sufficiency="insufficient",
                        safety_assessment="unsafe_overclaim", answer_grounding="unsupported")
    m = human_metrics(catalog, [review])
    assert m["false_confidence_rate"]["value"] == 1.
    assert m["refusal_recall"]["value"] == 0.
    review.evidence_sufficiency = None
    assert human_metrics(catalog, [review])["false_confidence_rate"]["value"] is None


def test_kappa_hand_calculation_and_undefined():
    assert agreement_stats([(1, 1), (1, 0), (0, 0), (0, 1)])["cohens_kappa"] == 0.
    assert agreement_stats([(1, 1), (0, 0)])["cohens_kappa"] == 1.
    assert agreement_stats([(1, 1), (1, 1)])["cohens_kappa"] is None
    assert agreement_stats([])["raw_agreement"] is None


def test_original_directory_protected(catalog):
    with pytest.raises(ValueError, match="separate directory"):
        ReviewStore(catalog.original_csv.parent, catalog)


def test_blank_draft_does_not_create_human_metrics(store, catalog):
    store.save(CaseReview(case_id="S01", reviewer_name="Alice"), 0)
    assert report(catalog, store)["human_reviewed_labels"] is None
    assert not catalog.complete(store.load("Alice", "S01")[0])


def test_stale_import_does_not_overwrite_newer_work(store, catalog):
    review = example(catalog)
    store.save(review, 0)
    exported = store.export_csv()
    latest, revision = store.load("Alice", "S01")
    latest.reviewer_notes = "Newer independent judgment"
    store.save(latest, revision)
    with pytest.raises(ReviewConflict, match="Stale import"):
        store.import_csv(exported, "Importer")
    saved, revision = store.load("Alice", "S01")
    assert saved.reviewer_notes == "Newer independent judgment"
    assert revision == 2
