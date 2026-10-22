"""Strict provider contract and backwards-compatible UI answer adapter."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .agent import MAINTENANCE_DISCLAIMER
from .models import AgentAnswer, ConfidenceAssessment, GroundedStatement

NO_CAUSE = "No explicit cause was documented in the retrieved historical records."
LIMITATION = "Historical actions are not verified successful repairs or approved procedures."
DIVERGENCE = "Historical records contain different action texts; no single action is endorsed."


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArguments(StrictModel):
    query: str = Field(min_length=1, max_length=10000)
    domain: Literal["aviation"]
    top_k: int = Field(ge=1, le=50)


class GetArguments(StrictModel):
    record_id: str = Field(min_length=1, max_length=128)


class SourceEvidence(StrictModel):
    record_id: str
    problem_original: str
    action_original: str


class EvidenceTrace(StrictModel):
    sequence: int
    call_id: str
    tool_name: Literal["search_records", "get_record"]
    arguments: SearchArguments | GetArguments
    started_at: datetime
    duration_ms: float
    status: Literal["success", "error"]
    result_record_ids: list[str]
    result_summary: str
    error: str | None


class CopilotAnswer(StrictModel):
    """All fields are required. Host verifies echoed provenance before rendering."""
    query: str
    normalized_query: str
    status: Literal["answered", "insufficient_evidence"]
    summary: list[GroundedStatement]
    similar_historical_findings: list[GroundedStatement]
    documented_causes: list[GroundedStatement]
    historical_actions: list[GroundedStatement]
    cited_record_ids: list[str]
    evidence: list[SourceEvidence]
    confidence: ConfidenceAssessment
    limitations: list[Literal[
        "Historical actions are not verified successful repairs or approved procedures.",
        "Historical records contain different action texts; no single action is endorsed.",
    ]]
    disclaimer: Literal[MAINTENANCE_DISCLAIMER]
    tool_trace: list[EvidenceTrace]


class ModeInfo(StrictModel):
    mode: Literal["llm", "fallback", "deterministic"]
    model: str | None
    llm_used: bool
    fallback: bool
    fallback_reason: str | None
    provider_requests: int
    rejected_claims: int
    generated_claims: int
    supported_claims: int
    generated_citation_pairs: int
    valid_id_pairs: int
    fetched_id_pairs: int
    prompt_version: str


class OptionalAgentAnswer(AgentAnswer):
    mode_info: ModeInfo
    structured_answer: CopilotAnswer | None = None


def strict_schema(model):
    schema = model.model_json_schema()
    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(schema)
    return schema
