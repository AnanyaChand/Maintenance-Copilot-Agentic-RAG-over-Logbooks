"""Explicit protocol simulator for tests/evaluation only. Never a runtime provider fallback."""
import json

from .llm_grounding import observation, limitations_for, explicit_causes
from .models import CanonicalRecord


class MockToolCallingProvider:
    def __init__(self):
        self.requests = 0

    def complete(self, payload, *, timeout):
        self.requests += 1
        messages = payload["messages"]
        context = json.loads(messages[-1]["content"].split(": ", 1)[1])
        user = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        if not results:
            name, args = "search_records", user
        else:
            eligible = [h["record_id"] for h in results[0]["results"] if h["evidence_eligible"]][:3]
            fetched = [r["record_id"] for r in results[1:]]
            pending = [rid for rid in eligible if rid not in fetched]
            if pending:
                name, args = "get_record", {"record_id": pending[0]}
            else:
                records = [CanonicalRecord.model_validate(r) for r in results[1:]]
                def claim(record, text):
                    return {"text": observation(record.record_id, text), "record_ids": [record.record_id]}
                issues = [claim(r, r.problem_original) for r in records]
                return {"content": json.dumps(dict(
                    query=user["query"], normalized_query=context["normalized_query"], status="answered",
                    summary=issues[:1], similar_historical_findings=issues,
                    documented_causes=[claim(r, text) for r in records for text in explicit_causes(r)],
                    historical_actions=[claim(r, r.action_original) for r in records],
                    cited_record_ids=[r.record_id for r in records],
                    evidence=[dict(record_id=r.record_id, problem_original=r.problem_original,
                                   action_original=r.action_original) for r in records],
                    confidence=context["confidence"], limitations=limitations_for(records),
                    disclaimer=context["disclaimer"], tool_trace=context["tool_trace"]))}
        return {"content": None, "tool_calls": [{"id": f"mock-call-{self.requests}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}
