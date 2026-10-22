import csv
import json
from pathlib import Path

import pytest

from maintenance_copilot.calibration import (EmpiricalUncertainty, bootstrap_intervals,
                                            family_split, threshold_grid)
from maintenance_copilot.evaluation import Case
from maintenance_copilot.selective import A, B, C, Thresholds, conflicts, decide
from maintenance_copilot.review import export_review, reviewed_metrics
from test_agent import context


def candidate(**updates):
    value = dict(record_id="R1", reranker=.999, fingerprint=.99, technical=1., agreement=1.,
                 corroboration=0., evidence_gate=.99, overlap=.99, similarity=.99, conflicts=[],
                 action_available=True, group="starter gear teeth missing", group_count=1,
                 action_variants=1, requested_technical=[], cross_logit=8.)
    value.update(updates)
    return value


def features(*candidates, blocked=False):
    return {"candidates": list(candidates), "intent_blocked": blocked}


def test_tier_a_strict_single_record_rescue():
    result = decide(features(candidate()), Thresholds())
    assert result["tier"] == A
    assert result["exact_match_rescue"]
    for update in (dict(reranker=.97), dict(overlap=.7), dict(action_available=False),
                   dict(similarity=.9), dict(technical=.5)):
        result = decide(features(candidate(**update)), Thresholds())
        assert not result["exact_match_rescue"]
        assert result["tier"] != A


def test_same_fault_divergent_actions_allowed_with_warning():
    result = decide(features(candidate(group_count=2, corroboration=.5, action_variants=2),
                             candidate(record_id="R2", group_count=2, corroboration=.5, action_variants=2)), Thresholds())
    assert result["tier"] == B
    assert result["selected_ids"] == ["R1", "R2"]
    assert result["action_divergence"]
    assert "Multiple historical action variants found; no single action is endorsed." in result["limitations"]


def test_weak_and_action_only_matches_refuse():
    assert decide(features(candidate(reranker=.1)), Thresholds())["tier"] == C
    assert decide(features(candidate(overlap=0.)), Thresholds())["tier"] == C
    assert decide(features(candidate()), Thresholds(evidence_gate=1.))["tier"] == C
    assert decide(features(candidate(), blocked=True), Thresholds())["selected_ids"] == []


def test_conflicting_side_cannot_enter_a_or_b():
    conflict = conflicts({"direction:left"}, {"direction:right"})
    assert conflict
    assert decide(features(candidate(conflicts=conflict)), Thresholds())["tier"] == C
    result = decide(features(candidate(), candidate(record_id="bad", conflicts=conflict)), Thresholds())
    assert result["tier"] == B
    assert not result["exact_match_rescue"]
    assert "bad" not in result["selected_ids"]


def test_family_split_and_grid_are_reproducible():
    payload = json.loads((Path(__file__).parents[1] / "evaluation/benchmark.v1.json").read_text())
    cases = [Case.model_validate(c) for c in payload["cases"]]
    split = family_split(cases)
    assert split == family_split(cases)
    reversed_split = family_split(list(reversed(cases)))
    assert {k: set(v) for k, v in split.items()} == {k: set(v) for k, v in reversed_split.items()}
    assert set(split["development"]).isdisjoint(split["held_out"])
    dev_families = {c.family for c in cases if c.id in split["development"]}
    held_families = {c.family for c in cases if c.id in split["held_out"]}
    assert not dev_families & held_families
    assert threshold_grid() == threshold_grid()
    assert len(threshold_grid()) == 64


def row(case_id, answered=True, correct=True):
    return {"id": case_id, "family": case_id, "status": "answered" if answered else "insufficient_evidence",
            "tier": B if answered else C, "expected_refusal": not correct,
            "citation_pairs": 2 if answered else 0, "relevant_pairs": 2 if correct and answered else 0,
            "grounded_pairs": 2 if answered else 0, "claims": 2 if answered else 0,
            "grounded_claims": 2 if answered else 0, "decision": {"nonconformity": .15}}


def test_bootstrap_intervals_and_tier_aware_uncertainty():
    rows = [row("one"), row("two", False), row("three", False, False), row("four", True, False)]
    intervals = bootstrap_intervals(rows, iterations=100)
    assert intervals == bootstrap_intervals(rows, iterations=100)
    assert all(v is None or 0 <= v[0] <= v[1] <= 1 for v in intervals.values())
    calibration = EmpiricalUncertainty.fit(rows)
    assert len(calibration.references["pooled"]) == 1  # unsafe answer excluded
    low = calibration.predict({"tier": B, "nonconformity": .01})
    high = calibration.predict({"tier": C, "nonconformity": 1.})
    assert low["score"] < high["score"]
    assert high["samples"] == 1
    assert len(high["interval"]) == 2
    assert EmpiricalUncertainty.fit([]).predict({"tier": C, "nonconformity": 1.})["interval"] == [0., 1.]


def test_review_no_human_labels_means_no_human_metrics(tmp_path):
    from types import SimpleNamespace
    original = tmp_path / "original.csv"
    report = {"cases": [{"id": "S1", "tier": B, "answer": {
        "query": "fault", "cited_record_ids": ["R1"], "answer_text": "Historical source quote"}}]}
    export_review(report, {"R1": SimpleNamespace(problem_original="fault", action_original="inspected")}, original)
    assert reviewed_metrics(original, original) is None
    with original.open() as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(review_origin="human", relevance="relevant", grounding="supported",
                   refusal="correct", reviewer="test-reviewer", reviewed_at="2026-10-22")
    reviewed = tmp_path / "reviewed.csv"
    with reviewed.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metrics = reviewed_metrics(reviewed, original)
    assert metrics["citation_relevance_partial_credit"] == 1.
    assert metrics["reviewed_cases"] == 1
    assert metrics["annotations"][0]["reviewer"] == "test-reviewer"


def test_real_sweep_reproducible_and_does_not_access_held_out():
    from maintenance_copilot.calibration import sweep
    from maintenance_copilot.ingestion import ingest_csv
    from maintenance_copilot.models import ConfidenceAssessment, QueryRewrite, SearchHit, SearchResponse
    records = ingest_csv(Path(__file__).parent / "fixtures/synthetic_maintnet.csv").records
    record = records[0]
    case = Case(id="dev", category="supported", query="baffle seal loose", family="dev-family",
                expected_relevant_record_ids=[record.record_id], relevance_grades={record.record_id: 3},
                rationale="Synthetic test only", expected_evidence_strength="exact", expected_refusal=False)
    hit = SearchHit(record_id=record.record_id, problem_original=record.problem_original,
                    action_original=record.action_original, rrf_rank=1, rrf_score=.03,
                    query_token_coverage=1.)
    response = SearchResponse(query=QueryRewrite(original_query=case.query, normalized_query=case.query),
        domain="aviation", top_k=5, status="supported", results=[hit],
        confidence=ConfidenceAssessment(score=.9, label="high", should_refuse=False, explanation=[],
            semantic_signal=1., lexical_signal=1., consensus_signal=1., separation_signal=1.))
    entry = features(candidate(record_id=record.record_id))
    entry.update(response=response.model_dump(mode="json"))
    class GuardedFeatures(dict):
        def __getitem__(self, key):
            assert key == "dev", "Held-out access during fitting"
            return super().__getitem__(key)
    cache = GuardedFeatures(dev=entry, held_out="poison: never read")
    store = {r.record_id: r for r in records}
    first = sweep([case], cache, store)
    assert first == sweep([case], cache, store)
    assert len(first[0]) == 64


def test_agent_tier_b_limitations_and_tier_c_exact_text(context):
    from maintenance_copilot.agent import MaintenanceAgent
    from maintenance_copilot.models import EvidenceAssessment
    from maintenance_copilot.retrieval import REFUSAL_MESSAGE
    from maintenance_copilot.selective import VARIANTS_WARNING
    records, response = context
    response = response.model_copy(update={"selective": EvidenceAssessment(
        tier=B, uncertainty=.4, uncertainty_interval=(.2, .6), calibration_samples=10,
        gates=["WARNING: action divergence"], limitations=[VARIANTS_WARNING], action_divergence=True)})
    agent = MaintenanceAgent(search_tool=lambda **kw: response,
                             get_record_tool={r.record_id: r for r in records}.__getitem__)
    answer = agent.answer("baffle fault")
    assert answer.answer_text.startswith("Historical evidence found with limitations")
    assert VARIANTS_WARNING in answer.answer_text
    assert answer.selective.tier == B
    assert {e.tool_name for e in answer.tool_trace} == {"search_records", "get_record"}
    assert not answer.possible_causes
    response = response.model_copy(update={"results": [], "status": "insufficient_evidence",
        "selective": response.selective.model_copy(update={"tier": C})})
    answer = agent.answer("unsupported")
    assert answer.answer_text == REFUSAL_MESSAGE
    assert not answer.evidence and not answer.cited_record_ids
    assert len(answer.tool_trace) == 1
