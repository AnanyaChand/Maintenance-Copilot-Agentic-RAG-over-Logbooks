import copy
import json
from pathlib import Path

import pytest

from maintenance_copilot.agent import MaintenanceAgent, MAINTENANCE_DISCLAIMER
from maintenance_copilot.ingestion import ingest_csv
from maintenance_copilot.llm_agent import answer_with_mode
from maintenance_copilot.llm_config import LLMConfig
from maintenance_copilot.llm_grounding import observation, limitations_for
from maintenance_copilot.llm_models import CopilotAnswer, strict_schema
from maintenance_copilot.models import ConfidenceAssessment, SearchResponse, SearchHit
from maintenance_copilot.query import normalize_query


@pytest.fixture
def context():
    records = ingest_csv(Path(__file__).parent / "fixtures/synthetic_maintnet.csv").records
    confidence = ConfidenceAssessment(score=.9, label="high", should_refuse=False,
        explanation=["test support"], semantic_signal=1, lexical_signal=1, consensus_signal=1, separation_signal=0)
    search = SearchResponse(query=normalize_query("baffle seal"), domain="aviation", top_k=5,
        status="supported", confidence=confidence, results=[SearchHit(
            record_id=r.record_id, problem_original=r.problem_original, action_original=r.action_original,
            rrf_score=.03, rrf_rank=i+1, query_token_coverage=1) for i, r in enumerate(records)])
    calls = []
    def retrieve(**args):
        calls.append(("search_records", args))
        return search
    def fetch(rid):
        calls.append(("get_record", {"record_id": rid}))
        return next(r for r in records if r.record_id == rid)
    return records, search, calls, MaintenanceAgent(search_tool=retrieve, get_record_tool=fetch)


def enabled():
    return LLMConfig.from_env(dict(COPILOT_MODE="llm", LLM_ENABLED="true", LLM_BASE_URL="https://provider.example/v1",
                                  LLM_API_KEY="test-only-secret", LLM_MODEL="configured-model", LLM_TIMEOUT_SECONDS="30"))


class ScriptedProvider:
    """Emulates the wire protocol; never bypasses the application's real tools."""
    def __init__(self, records, mutation=None, fail=None):
        self.records, self.mutation, self.fail = records, mutation, fail
        self.calls = []

    def complete(self, payload, *, timeout):
        self.calls.append(copy.deepcopy(payload))
        if self.fail:
            raise self.fail
        context = json.loads(payload["messages"][-1]["content"].split(": ", 1)[1])
        if context["stage"] == "search":
            name, args = "search_records", {"query": "baffle seal", "domain": "aviation", "top_k": 5}
        elif not context["fetched_ids"]:
            name, args = "get_record", {"record_id": self.records[0].record_id}
        else:
            r = self.records[0]
            claim = lambda text: {"text": observation(r.record_id, text), "record_ids": [r.record_id]}
            answer = dict(query=context["query"], normalized_query=context["normalized_query"], status="answered",
                          summary=[claim(r.problem_original)], similar_historical_findings=[claim(r.problem_original)],
                          documented_causes=[], historical_actions=[claim(r.action_original)],
                          cited_record_ids=[r.record_id], evidence=[dict(record_id=r.record_id,
                            problem_original=r.problem_original, action_original=r.action_original)],
                          confidence=context["confidence"], limitations=limitations_for([r]),
                          disclaimer=MAINTENANCE_DISCLAIMER, tool_trace=context["tool_trace"])
            if self.mutation == "malformed":
                return {"content": "not JSON with test-only-secret"}
            if self.mutation:
                self.mutation(answer)
            return {"content": json.dumps(answer)}
        return {"content": None, "tool_calls": [{"id": f"call-{len(self.calls)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


def run(context, provider, config=None):
    return answer_with_mode("baffle seal", agent=context[3], config=config or enabled(), provider=provider)


def test_default_no_key_is_unchanged_deterministic(context):
    provider = ScriptedProvider(context[0], fail=AssertionError("must not run"))
    answer = run(context, provider, LLMConfig.from_env({}))
    assert answer.provider == "deterministic-extractive"
    assert not hasattr(answer, "mode_info")
    assert not provider.calls
    assert [c[0] for c in context[2]] == ["search_records", "get_record", "get_record"]


def test_requested_without_key_falls_back_without_network(context):
    provider = ScriptedProvider(context[0])
    answer = run(context, provider, LLMConfig.from_env({"COPILOT_MODE": "llm", "LLM_ENABLED": "true"}))
    assert answer.mode_info.fallback and not answer.mode_info.llm_used
    assert not provider.calls
    assert "missing" in answer.mode_info.fallback_reason


@pytest.mark.parametrize("failure", [RuntimeError("401 key=test-only-secret endpoint=https://secret.example"), TimeoutError("secret timeout")])
def test_provider_error_or_timeout_falls_back_without_leaks(context, failure):
    answer = run(context, ScriptedProvider(context[0], fail=failure))
    assert answer.mode_info.fallback and answer.provider == "deterministic-extractive"
    public = answer.model_dump_json()
    assert "test-only-secret" not in public and "secret.example" not in public
    assert any(e.tool_name == "llm_request" and e.status == "error" for e in answer.tool_trace)
    assert any(e.tool_name == "llm_fallback" for e in answer.tool_trace)


def test_successful_llm_sequence_and_real_trace(context):
    provider = ScriptedProvider(context[0])
    answer = run(context, provider)
    assert answer.mode_info.mode == "llm", answer.mode_info.fallback_reason
    assert answer.mode_info.provider_requests == 3
    assert answer.cited_record_ids == ["T-001"]
    assert answer.structured_answer is not None
    evidence_events = [e for e in answer.tool_trace if e.tool_name in {"search_records", "get_record"}]
    assert [(e.tool_name, e.arguments) for e in evidence_events] == context[2]
    assert [e.tool_name for e in evidence_events] == ["search_records", "search_records", "get_record"]
    assert len([e for e in answer.tool_trace if e.tool_name == "llm_request"]) == len(provider.calls)
    assert [e.sequence for e in answer.tool_trace] == list(range(1, len(answer.tool_trace)+1))
    assert answer.mode_info.generated_claims == answer.mode_info.supported_claims == 3
    assert "No explicit cause was documented in the retrieved historical records." in answer.answer_text


def test_malformed_json_fallback(context):
    answer = run(context, ScriptedProvider(context[0], mutation="malformed"))
    assert answer.mode_info.fallback
    assert "invalid structured" in answer.mode_info.fallback_reason
    assert "test-only-secret" not in answer.model_dump_json()


@pytest.mark.parametrize("mutation", [
    lambda a: a["summary"][0].update(record_ids=["T-002"]),  # returned but never fetched
    lambda a: a["summary"][0].update(record_ids=["unknown"]),
    lambda a: a["summary"][0].update(record_ids=[]),
    lambda a: a["summary"][0].update(text="You should replace the seal."),
    lambda a: a["historical_actions"][0].update(text="Historical record [T-001] reports: “Engine repaired successfully.”"),
    lambda a: a["summary"].append({"text": "A faulty battery caused this.", "record_ids": ["T-001"]}),
    lambda a: a["confidence"].update(score=1.),
    lambda a: a["tool_trace"][0].update(result_record_ids=["invented"]),
    lambda a: a.update(limitations=["Guaranteed repair"]),
    lambda a: a.update(cited_record_ids=["T-001", "T-001"]),
])
def test_unfetched_uncited_unsupported_procedural_or_fabricated_output_rejected(context, mutation):
    answer = run(context, ScriptedProvider(context[0], mutation=mutation))
    assert answer.mode_info.fallback
    assert answer.provider == "deterministic-extractive"
    assert answer.structured_answer is None


def test_refusal_never_calls_llm_and_keeps_baseline_trace(context):
    records, search, calls, agent = context
    refused = search.model_copy(update={"status": "insufficient_evidence"})
    agent.search_tool = lambda **kw: refused
    provider = ScriptedProvider(records, fail=AssertionError("no API on refusal"))
    answer = run(context, provider)
    assert answer.answer_text == "Insufficient evidence in the available maintenance records"
    assert answer.cited_record_ids == [] and len(answer.tool_trace) == 1
    assert answer.tool_trace[0].tool_name == "search_records"
    assert not provider.calls and not calls


def test_model_cannot_skip_search(context):
    class Direct:
        def complete(self, *a, **kw):
            return {"content": "I know the repair."}
    answer = run(context, Direct())
    assert answer.mode_info.fallback
    assert "tool sequence" in answer.mode_info.fallback_reason


@pytest.mark.parametrize("name,args", [("search_records", {"query": "different query", "domain": "aviation", "top_k": 5}),
                                       ("get_record", {"record_id": "unknown"})])
def test_model_cannot_rewrite_search_or_fetch_before_search(context, name, args):
    class BadTool:
        def complete(self, *a, **kw):
            return {"tool_calls": [{"id": "bad-call", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]}
    answer = run(context, BadTool())
    assert answer.mode_info.fallback
    assert all(call[1].get("query", "baffle seal") == "baffle seal" for call in context[2])
    assert all(call[1].get("record_id", "T-001") != "unknown" for call in context[2])


def test_verifier_counts_unsupported_claims(context):
    def mutation(a):
        a["summary"][0]["text"] = "A new component would fix this."
    answer = run(context, ScriptedProvider(context[0], mutation=mutation))
    assert answer.mode_info.generated_claims == 3
    assert answer.mode_info.rejected_claims == 1
    assert answer.mode_info.supported_claims == 2


@pytest.mark.parametrize("change", [{"LLM_TIMEOUT_SECONDS": "nan"}, {"LLM_BASE_URL": "https://user:secret@example.com/v1"},
    {"LLM_BASE_URL": "https://example.com/v1?key=secret"}, {"LLM_BASE_URL": "http://remote.example/v1"},
    {"LLM_MODEL": "test-only-secret"}, {"COPILOT_MODE": "typo"}])
def test_config_validation_and_secret_repr(change):
    env = dict(COPILOT_MODE="llm", LLM_ENABLED="true", LLM_API_KEY="test-only-secret",
               LLM_BASE_URL="https://provider.example/v1", LLM_MODEL="configured-model")
    config = LLMConfig.from_env({**env, **change})
    assert not config.enabled and config.problem == "invalid_configuration"
    assert "secret" not in repr(config)
    assert "test-only-secret" not in repr(enabled())


def test_strict_schema_has_no_open_objects_or_optional_properties():
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)
    check(strict_schema(CopilotAnswer))


def test_changed_second_retrieval_cannot_be_overridden(context):
    calls = 0
    original = context[3].search_tool
    def changing(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        return result if calls == 1 else result.model_copy(update={"status": "insufficient_evidence"})
    context[3].search_tool = changing
    answer = run(context, ScriptedProvider(context[0]))
    assert answer.status == "insufficient_evidence"
    assert not any(t.tool_name == "get_record" for t in answer.tool_trace)


@pytest.mark.parametrize("mutation, expected", [(None, "LLM Tool-Calling Mode"), ("malformed", "LLM fallback → Deterministic Evidence Mode")])
def test_streamlit_mode_and_trace_use_real_workflow(context, monkeypatch, mutation, expected):
    from streamlit.testing.v1 import AppTest
    import maintenance_copilot.agent as agent_module
    monkeypatch.setenv("COPILOT_MODE", "deterministic")
    provider = ScriptedProvider(context[0], mutation=mutation)
    monkeypatch.setattr(agent_module, "answer_query", lambda query: run(context, provider))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py")).run()
    app.text_area[0].set_value("baffle seal")
    next(b for b in app.button if b.label == "Analyze historical records").click().run()
    assert not app.exception
    captions = "\n".join(c.value for c in app.caption)
    assert expected in captions
    assert "test-only-secret" not in captions
    assert len(provider.calls) == 3
    trace = app.session_state["answer"].tool_trace
    assert len([e for e in trace if e.tool_name == "llm_request"]) == 3


def test_timeout_transport_is_bounded_and_credentials_not_in_command(monkeypatch):
    import subprocess
    from maintenance_copilot.llm_transport import OpenAICompatibleProvider, ProviderFailure
    def run_process(command, **kwargs):
        assert "test-only-secret" not in " ".join(command)
        assert kwargs["timeout"] == .1
        assert kwargs["capture_output"] is True
        raise subprocess.TimeoutExpired(command, .1, output="secret payload")
    monkeypatch.setattr(subprocess, "run", run_process)
    with pytest.raises(ProviderFailure, match="^timeout$"):
        OpenAICompatibleProvider(enabled()).complete({}, timeout=.1)


@pytest.mark.parametrize("content", ['{"error":"raw-secret-provider-error"}', 'invalid secret JSON'])
def test_transport_never_propagates_worker_errors(monkeypatch, content):
    import subprocess
    from types import SimpleNamespace
    from maintenance_copilot.llm_transport import OpenAICompatibleProvider, ProviderFailure
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=content))
    with pytest.raises(ProviderFailure) as error:
        OpenAICompatibleProvider(enabled()).complete({}, timeout=1)
    assert str(error.value) == "provider_error"


def test_http_worker_sends_strict_api_request_without_following_redirects(monkeypatch, capsys):
    import io
    from urllib import error
    from maintenance_copilot import llm_http_worker as worker
    job = dict(url="https://provider.example/v1/chat/completions", key="secret-test-key", timeout=1,
               payload={"model": "configured-model", "messages": []})
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps(job)))
    class Opener:
        def open(self, req, **kwargs):
            assert req.headers["Authorization"] == "Bearer secret-test-key"
            assert json.loads(req.data)["model"] == "configured-model"
            raise error.HTTPError(req.full_url, 401, "secret provider body", {}, None)
    def build(handler):
        assert isinstance(handler, worker.NoRedirect)
        assert handler.redirect_request(None, None, 302, "", {}, "https://other.example") is None
        return Opener()
    monkeypatch.setattr(worker.request, "build_opener", build)
    worker.main()
    assert json.loads(capsys.readouterr().out) == {"error": "provider_error"}
