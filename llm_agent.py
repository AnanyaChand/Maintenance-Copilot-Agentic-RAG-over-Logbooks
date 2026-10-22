"""Optional bounded tool-calling workflow. Local retrieval always gates API access."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

from pydantic import ValidationError

from .agent import MaintenanceAgent, MAINTENANCE_DISCLAIMER, AgentExecutionError
from .llm_config import LLMConfig, REASONS
from .llm_grounding import verify_answer, VerificationFailure, render_verified
from .llm_models import (CopilotAnswer, SearchArguments, GetArguments, ModeInfo,
                         OptionalAgentAnswer, strict_schema, LIMITATION, DIVERGENCE)
from .llm_transport import OpenAICompatibleProvider, ProviderFailure
from .models import SearchResponse, ToolTraceEvent, AgentAnswer

PROMPT_VERSION = "historical-evidence-v1"
PROMPT_PATH = Path(__file__).with_name("resources") / "llm_system_v1.txt"


def tool_schema(name, model, description):
    return {"type": "function", "function": {"name": name, "description": description,
            "strict": True, "parameters": strict_schema(model)}}


SEARCH_TOOL = tool_schema("search_records", SearchArguments, "Search historical aviation records for the unchanged user query.")
GET_TOOL = tool_schema("get_record", GetArguments, "Fetch a full eligible record returned by search_records.")


def event(trace, name, *, summary, status="success", arguments=None, started=None, began=None):
    trace.append(ToolTraceEvent(
        sequence=len(trace) + 1, call_id=str(uuid4()), tool_name=name,
        arguments=arguments or {}, started_at=started or datetime.now(timezone.utc),
        duration_ms=max(0., (time.perf_counter() - began) * 1000) if began is not None else 0.,
        status=status, result_summary=summary))


def answer_with_mode(query, domain="aviation", top_k=5, *, config=None, provider=None, agent=None):
    config = config if config is not None else LLMConfig.from_env()
    agent = agent or MaintenanceAgent()
    if not config.requested:
        return agent.answer(query, domain, top_k)

    # Authoritative local gate happens before any provider is constructed/called.
    trace = []
    initial, search_event = agent._trace_call(
        sequence=1, tool_name="search_records",
        arguments={"query": query, "domain": domain, "top_k": top_k}, trace=trace,
        call=lambda: agent.search_tool(query=query, domain=domain, top_k=top_k))
    trace.append(search_event)
    if (initial.status == "insufficient_evidence" or initial.confidence.should_refuse
            or not any(hit.evidence_eligible for hit in initial.results)):
        # Same text, provider, fields and single real search event as baseline refusal.
        return agent.answer_from_search(query, initial, trace)

    requests = 0
    audit = dict(rejected_claims=0, generated_claims=0, supported_claims=0,
                 generated_citation_pairs=0, valid_id_pairs=0, fetched_id_pairs=0)

    def mode_info(mode, reason=None):
        return ModeInfo(mode=mode, model=config.model if config.enabled else None,
                        llm_used=requests > 0, fallback=mode == "fallback", fallback_reason=reason,
                        provider_requests=requests, prompt_version=PROMPT_VERSION, **audit)

    def fallback(code):
        reason = REASONS.get(code, REASONS["provider_error"])
        event(trace, "llm_fallback", summary=reason, arguments={"reason_code": code if code in REASONS else "provider_error"})
        result = agent.answer_from_search(query, initial, trace)
        return OptionalAgentAnswer(**result.model_dump(), mode_info=mode_info("fallback", reason))

    if not config.enabled or config.problem:
        return fallback(config.problem or "invalid_configuration")

    try:
        provider = provider or OpenAICompatibleProvider(config)
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        messages = [{"role": "user", "content": json.dumps({"query": query, "domain": domain, "top_k": top_k})}]
        searched = None
        fetched = {}
        evidence_trace = []
        tool_call_ids = set()
        deadline = time.monotonic() + config.timeout
        # Search, up to three fetches, final response. No unbounded agent loop.
        for _ in range(5):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderFailure("timeout")
            stage = "search" if searched is None else "fetch_or_answer" if len(fetched) < 3 else "answer"
            context = {"query": query, "normalized_query": initial.query.normalized_query,
                       "confidence": initial.confidence.model_dump(mode="json"),
                       "disclaimer": MAINTENANCE_DISCLAIMER, "tool_trace": evidence_trace,
                       "allowed_limitations": [LIMITATION, DIVERGENCE],
                       "stage": stage, "fetched_ids": list(fetched)}
            payload = {"model": config.model,
                       "messages": [{"role": "system", "content": prompt}, *messages,
                                    {"role": "system", "content": "Host provenance (copy exactly where applicable): " + json.dumps(context)}],
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "CopilotAnswer", "strict": True, "schema": strict_schema(CopilotAnswer)}},
                       "max_completion_tokens": 6000}
            if stage != "answer":
                payload.update(tools=[SEARCH_TOOL] if stage == "search" else [GET_TOOL],
                               parallel_tool_calls=False,
                               tool_choice={"type": "function", "function": {"name": "search_records"}}
                               if stage == "search" else "required" if not fetched else "auto")
            started, began = datetime.now(timezone.utc), time.perf_counter()
            requests += 1
            try:
                message = provider.complete(payload, timeout=remaining)
                if time.monotonic() > deadline:
                    raise ProviderFailure("timeout")
            except Exception as exc:
                code = exc.code if isinstance(exc, ProviderFailure) and exc.code in REASONS else "timeout" if isinstance(exc, TimeoutError) else "provider_error"
                event(trace, "llm_request", summary=REASONS[code], status="error",
                      arguments={"stage": stage}, started=started, began=began)
                raise ProviderFailure(code) from None
            event(trace, "llm_request", summary="Provider response received; content is untrusted until validated.",
                  arguments={"stage": stage}, started=started, began=began)
            if not isinstance(message, dict) or message.get("refusal"):
                raise ProviderFailure("malformed_output")
            calls = message.get("tool_calls")
            if calls:
                if (stage == "answer" or not isinstance(calls, list) or len(calls) != 1
                        or message.get("content") not in (None, "")):
                    raise ProviderFailure("invalid_tool_sequence")
                call = calls[0]
                call_id = call.get("id")
                if not isinstance(call_id, str) or not 1 <= len(call_id) <= 128 or call_id in tool_call_ids or call.get("type") != "function":
                    raise ProviderFailure("invalid_tool_sequence")
                tool_call_ids.add(call_id)
                function = call.get("function", {})
                name = function.get("name")
                model = SearchArguments if stage == "search" else GetArguments
                if name != ("search_records" if stage == "search" else "get_record"):
                    raise ProviderFailure("invalid_tool_sequence")
                try:
                    args = model.model_validate_json(function["arguments"])
                except (ValidationError, ValueError, TypeError, KeyError):
                    raise ProviderFailure("invalid_tool_sequence") from None
                if stage == "search":
                    if args.model_dump() != {"query": query, "domain": domain, "top_k": top_k}:
                        raise ProviderFailure("invalid_tool_sequence")
                    invoke = lambda: agent.search_tool(**args.model_dump())
                else:
                    allowed = {h.record_id: h for h in searched.results if h.evidence_eligible}
                    if args.record_id not in allowed or args.record_id in fetched:
                        raise ProviderFailure("invalid_tool_sequence")
                    invoke = lambda: agent.get_record_tool(args.record_id)
                tool_started, tool_began = datetime.now(timezone.utc), time.perf_counter()
                try:
                    result, actual = agent._trace_call(sequence=len(trace)+1, tool_name=name,
                        arguments=args.model_dump(), trace=trace, call=invoke)
                except AgentExecutionError as exc:
                    trace.append(exc.tool_trace[-1].model_copy(update={"error": None, "result_summary": "Evidence tool failed."}))
                    raise ProviderFailure("tool_error") from None
                except Exception:
                    event(trace, name, summary="Evidence tool failed.", status="error", arguments=args.model_dump(),
                          started=tool_started, began=tool_began)
                    raise ProviderFailure("tool_error") from None
                trace.append(actual)
                evidence_trace.append(actual.model_dump(mode="json"))
                if stage == "search":
                    # A second actual search executes the provider's request. No gate bypass,
                    # rewritten query, changed corpus or changed result can enter synthesis.
                    if isinstance(result, SearchResponse) and (result.status == "insufficient_evidence"
                            or result.confidence.should_refuse or not any(h.evidence_eligible for h in result.results)):
                        initial = result
                        return fallback("retrieval_changed")
                    if not isinstance(result, SearchResponse) or result != initial:
                        raise ProviderFailure("metadata_mismatch")
                    searched = result
                else:
                    hit = allowed[args.record_id]
                    if (result.record_id != args.record_id or result.problem_original != hit.problem_original
                            or result.action_original != hit.action_original):
                        raise ProviderFailure("metadata_mismatch")
                    fetched[result.record_id] = result
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": function["arguments"]}}]})
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result.model_dump_json()})
                continue
            if searched is None or not fetched:
                raise ProviderFailure("invalid_tool_sequence")
            try:
                answer = CopilotAnswer.model_validate_json(message.get("content"), strict=True)
            except (ValidationError, ValueError, TypeError):
                raise ProviderFailure("malformed_output") from None
            try:
                # Membership in actual search results is stricter than corpus existence.
                audit.update(verify_answer(answer, query=query, search=searched, fetched=fetched,
                    evidence_trace=evidence_trace, known_ids={h.record_id for h in searched.results}))
            except VerificationFailure as exc:
                audit.update(exc.audit)
                event(trace, "llm_validation", summary=REASONS[exc.code], status="error")
                return fallback(exc.code)
            event(trace, "llm_validation", summary="All claims, citations and echoed provenance passed deterministic verification.")
            return OptionalAgentAnswer(
                query=query, normalized_query=initial.query.normalized_query, query_rewrite=initial.query,
                status="answered", answer_text=render_verified(answer),
                observed_issue_patterns=answer.similar_historical_findings, possible_causes=answer.documented_causes,
                historical_actions=answer.historical_actions, cited_record_ids=answer.cited_record_ids,
                evidence=[h for rid in answer.cited_record_ids for h in searched.results if h.record_id == rid],
                confidence=initial.confidence, provider="openai-compatible-tool-calling",
                tool_trace=trace, disclaimer=MAINTENANCE_DISCLAIMER, mode_info=mode_info("llm"),
                structured_answer=answer)
        raise ProviderFailure("invalid_tool_sequence")
    except ProviderFailure as exc:
        return fallback(exc.code)
    except Exception:
        # Untrusted provider structure must not leak details or crash the UI.
        return fallback("malformed_output")
