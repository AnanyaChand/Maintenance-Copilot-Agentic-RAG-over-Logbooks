import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from maintenance_copilot.evaluation import (
    Case, citation_counts, ranking_metrics, safety_metrics, sha, validate_suite, write_new,
)

SUITE = Path(__file__).parents[1] / "evaluation/benchmark.v1.json"


def test_frozen_suite_counts_labels_and_checksum():
    assert sha(SUITE) == "7bc9706f498aedaf6356f5802d3d7f7cd2977e228e464f091080710675f7e1df"
    payload = json.loads(SUITE.read_text())
    cases = validate_suite(payload, payload["evidence_snapshots"])
    assert len(cases) == 75
    assert sum(not c.expected_refusal for c in cases) == 55
    assert len({c.family for c in cases if not c.expected_refusal}) == 25
    assert payload["human_review_status"] == "pending"
    assert all(s["problem"] and s["action"] for s in payload["evidence_snapshots"].values())


def test_metric_hand_calculation_and_cutoffs():
    m = ranking_metrics(["x", "b", "a"], {"a": 3, "b": 1})
    assert m["Hit@1"] == 0
    assert m["Hit@3"] == m["Hit@5"] == 1
    assert m["MRR@50"] == .5
    assert m["nDCG@5"] == pytest.approx((1 / math.log2(3) + 7 / 2) / (7 + 1 / math.log2(3)))
    assert ranking_metrics([str(i) for i in range(51)], {"50": 1})["MRR@50"] == 0
    assert all(v == 0 for v in ranking_metrics([], {"a": 1}).values())
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], {"a": 1})


def test_refusal_confusion_matrix_and_empty_denominators():
    rows = [dict(expected_refusal=True, status="insufficient_evidence"),
            dict(expected_refusal=True, status="answered"),
            dict(expected_refusal=True, status="error"),
            dict(expected_refusal=False, status="insufficient_evidence"),
            dict(expected_refusal=False, status="answered")]
    m = safety_metrics(rows)
    assert m["refusal_precision"] == .5
    assert m["refusal_recall"] == pytest.approx(1 / 3)
    assert m["false_confidence_rate"] == pytest.approx(1 / 3)
    assert m["supported_answer_rate"] == .5
    assert m["citation_precision"] is None  # no vacuous 100% on all-refusal runs
    assert m["citation_coverage"] is None
    assert safety_metrics([])["refusal_recall"] is None


def test_inconsistent_label_rejected():
    row = json.loads(SUITE.read_text())["cases"][0]
    row["expected_refusal"] = True
    with pytest.raises(ValidationError):
        Case.model_validate(row)


def test_unknown_source_id_rejected():
    payload = json.loads(SUITE.read_text())
    with pytest.raises(ValueError, match="Unknown record"):
        validate_suite(payload, {})


def test_append_only_artifacts(tmp_path):
    output = tmp_path / "run.json"
    write_new(output, {"version": 1})
    with pytest.raises(FileExistsError):
        write_new(output, {"version": 2})
    assert json.loads(output.read_text()) == {"version": 1}


def test_citation_metrics_require_complete_fetched_and_searched_evidence():
    ns = SimpleNamespace
    record = ns(problem_original="NO LEAK FOUND", action_original="INSPECTED")
    answer = ns(
        tool_trace=[ns(tool_name="search_records", status="success", result_record_ids=["a"]),
                    ns(tool_name="get_record", status="success", arguments={"record_id": "a"})],
        observed_issue_patterns=[ns(text="LEAK FOUND", record_ids=["a"])],
        historical_actions=[ns(text="INSPECTED", record_ids=["a"])], possible_causes=[])
    counts = citation_counts(answer, {"a": record})
    assert counts == dict(citation_pairs=2, grounded_pairs=1, claims=2, grounded_claims=1)
    answer.tool_trace.pop()
    assert citation_counts(answer, {"a": record})["grounded_pairs"] == 0
