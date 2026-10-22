import pytest

from maintenance_copilot.agent import MaintenanceAgent, AgentExecutionError, GroundingError, validate_draft
from maintenance_copilot.ingestion import ingest_csv
from maintenance_copilot.models import ConfidenceAssessment, SearchResponse, SearchHit
from maintenance_copilot.providers import DeterministicEvidenceProvider
from maintenance_copilot.query import normalize_query
from pathlib import Path


@pytest.fixture
def context():
    records = ingest_csv(Path(__file__).parent / "fixtures/synthetic_maintnet.csv").records
    confidence = ConfidenceAssessment(score=.9, label="high", should_refuse=False,
        explanation=["test support"], semantic_signal=1, lexical_signal=1,
        consensus_signal=1, separation_signal=0)
    response = SearchResponse(query=normalize_query("baffle seal"), domain="aviation", top_k=5,
        status="supported", confidence=confidence, results=[SearchHit(
            record_id=r.record_id, problem_original=r.problem_original,
            action_original=r.action_original, rrf_score=.03, rrf_rank=i+1,
            query_token_coverage=1) for i, r in enumerate(records)])
    return records, response


def test_success_calls_both_tools_and_cites_fetched_evidence(context):
    records, response = context
    calls = []
    def search(**kwargs):
        calls.append(("search", kwargs))
        return response
    def fetch(record_id):
        calls.append(("get", record_id))
        return next(r for r in records if r.record_id == record_id)
    answer = MaintenanceAgent(search_tool=search, get_record_tool=fetch).answer("baffle seal")
    assert answer.status == "answered"
    assert [c[0] for c in calls] == ["search", "get", "get"]
    assert answer.cited_record_ids == [r.record_id for r in records]
    assert [e.sequence for e in answer.tool_trace] == [1, 2, 3]
    for r in records:
        assert f"[Record {r.record_id}]" in answer.answer_text
        assert r.action_original in answer.answer_text
    assert not answer.possible_causes


def test_refusal_never_fetches_or_synthesizes(context):
    _, response = context
    response = response.model_copy(update={"status": "insufficient_evidence",
        "confidence": response.confidence.model_copy(update={"should_refuse": True})})
    def forbidden(*args):
        pytest.fail("Should stop at retrieval refusal")
    class Provider:
        name = "unused"
        synthesize = forbidden
    answer = MaintenanceAgent(search_tool=lambda **kw: response,
        get_record_tool=forbidden, provider=Provider()).answer("unsupported")
    assert answer.answer_text == "Insufficient evidence in the available maintenance records"
    assert answer.cited_record_ids == []
    assert len(answer.tool_trace) == 1


@pytest.mark.parametrize("mutation", ["invented", "extra_citation", "partial", "cause", "unknown_id"])
def test_invalid_provider_output_is_blocked(context, mutation):
    records, _ = context
    draft = DeterministicEvidenceProvider().synthesize("query", records)
    if mutation == "invented":
        draft.historical_actions[0] = draft.historical_actions[0].model_copy(update={"text": "Replace entire engine"})
    elif mutation == "extra_citation":
        draft.historical_actions[0].record_ids.append(records[1].record_id)
    elif mutation == "partial":
        draft.historical_actions[0] = draft.historical_actions[0].model_copy(update={"text": "seal"})
    elif mutation == "cause":
        draft.possible_causes.append(draft.observed_issue_patterns[0])
    else:
        draft.selected_record_ids.append("unknown")
    with pytest.raises(GroundingError):
        validate_draft(draft, searched_ids={r.record_id for r in records},
            fetched_records={r.record_id: r for r in records})


def test_tool_failure_retains_prior_trace(context):
    _, response = context
    def fail(record_id):
        raise RuntimeError("record store unavailable")
    with pytest.raises(AgentExecutionError) as exc:
        MaintenanceAgent(search_tool=lambda **kw: response, get_record_tool=fail).answer("seal")
    assert [e.status for e in exc.value.tool_trace] == ["success", "error"]
    assert exc.value.tool_trace[-1].arguments == {"record_id": "T-001"}


def test_mismatched_fetch_is_blocked(context):
    records, response = context
    with pytest.raises(GroundingError):
        MaintenanceAgent(search_tool=lambda **kw: response,
            get_record_tool=lambda rid: records[1]).answer("seal")
