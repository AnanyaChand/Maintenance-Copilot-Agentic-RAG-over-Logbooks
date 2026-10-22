"""Validated data contracts for MaintNet ingestion."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CanonicalRecord(BaseModel):
    """A single aviation maintenance record with original and normalized text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1)
    domain: Literal["aviation"] = "aviation"
    problem_original: str = Field(min_length=1)
    action_original: str = Field(min_length=1)
    problem_normalized: str = Field(min_length=1)
    action_normalized: str = Field(min_length=1)
    ata_chapter: str | None = None
    opened_at: date | None = None
    closed_at: date | None = None
    source_row: int = Field(ge=2)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("record_id", "problem_normalized", "action_normalized")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("problem_original", "action_original")
    @classmethod
    def preserve_required_original_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("ata_chapter")
    @classmethod
    def blank_optional_text_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class IngestionReport(BaseModel):
    """Auditable facts about one ingestion run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str
    source_sha256: str
    encoding: str
    detected_columns: dict[str, str]
    source_rows: int = Field(ge=0)
    loaded_records: int = Field(ge=0)
    skipped_blank_rows: int = Field(ge=0)
    duplicate_ids: list[str] = Field(default_factory=list)


class RewriteChange(BaseModel):
    """One exact, auditable query normalization change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original: str
    replacement: str
    occurrences: int = Field(ge=1)


class QueryRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str
    normalized_query: str
    expansions_applied: list[RewriteChange] = Field(default_factory=list)
    typo_corrections_applied: list[RewriteChange] = Field(default_factory=list)


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    problem_original: str
    action_original: str
    bm25_score: float | None = None
    bm25_rank: int | None = Field(default=None, ge=1)
    vector_similarity: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = Field(ge=0)
    rrf_rank: int = Field(ge=1)
    query_token_coverage: float = Field(ge=0, le=1)
    match_explanation: list[str] = Field(default_factory=list)
    field_bm25_scores: dict[str, float] = Field(default_factory=dict)
    field_contributions: dict[str, float] = Field(default_factory=dict)
    problem_token_coverage: float | None = Field(default=None, ge=0, le=1)
    technical_matches: list[str] = Field(default_factory=list)
    technical_missing: list[str] = Field(default_factory=list)
    evidence_eligible: bool = True
    evidence_gate_reasons: list[str] = Field(default_factory=list)
    final_score: float | None = None
    final_rank: int | None = Field(default=None, ge=1)


class ConfidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0, le=1)
    label: Literal["high", "medium", "low"]
    should_refuse: bool
    explanation: list[str]
    semantic_signal: float = Field(ge=0, le=1)
    lexical_signal: float = Field(ge=0, le=1)
    consensus_signal: float = Field(ge=0, le=1)
    separation_signal: float = Field(ge=0, le=1)


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tier: Literal["TIER_A_STRONG_EVIDENCE", "TIER_B_SUPPORTED_WITH_LIMITATIONS", "TIER_C_INSUFFICIENT_EVIDENCE"]
    uncertainty: float = Field(ge=0, le=1)
    uncertainty_interval: tuple[float, float]
    calibration_samples: int = Field(ge=0)
    gates: list[str]
    limitations: list[str]
    action_divergence: bool = False
    exact_match_rescue: bool = False


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: QueryRewrite
    domain: Literal["aviation"]
    top_k: int = Field(ge=1, le=50)
    status: Literal["supported", "insufficient_evidence"]
    refusal_message: str | None = None
    confidence: ConfidenceAssessment
    results: list[SearchHit]
    selective: EvidenceAssessment | None = None


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index_version: str
    normalization_version: str
    embedding_model: str
    source_sha256: str
    record_count: int = Field(ge=1)
    embedding_dimensions: int = Field(ge=1)
    rrf_k: int = Field(ge=1)
    bm25_pool_size: int = Field(ge=1)
    vector_pool_size: int = Field(ge=1)
    abbreviation_sha256: str
    typo_corrections_sha256: str


class GroundedStatement(BaseModel):
    """An extractive statement tied to one or more fetched records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    record_ids: list[str] = Field(min_length=1)


class SynthesisDraft(BaseModel):
    """Provider-neutral evidence selection; validated before rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_record_ids: list[str] = Field(min_length=1)
    observed_issue_patterns: list[GroundedStatement]
    possible_causes: list[GroundedStatement] = Field(default_factory=list)
    historical_actions: list[GroundedStatement]


class ToolTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    call_id: str
    tool_name: Literal["search_records", "get_record", "llm_request", "llm_validation", "llm_fallback"]
    arguments: dict[str, str | int]
    started_at: datetime
    duration_ms: float = Field(ge=0)
    status: Literal["success", "error"]
    result_record_ids: list[str] = Field(default_factory=list)
    result_summary: str
    error: str | None = None


class AgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    normalized_query: str
    query_rewrite: QueryRewrite
    status: Literal["answered", "insufficient_evidence"]
    answer_text: str
    observed_issue_patterns: list[GroundedStatement] = Field(default_factory=list)
    possible_causes: list[GroundedStatement] = Field(default_factory=list)
    historical_actions: list[GroundedStatement] = Field(default_factory=list)
    cited_record_ids: list[str] = Field(default_factory=list)
    evidence: list[SearchHit] = Field(default_factory=list)
    confidence: ConfidenceAssessment
    provider: str
    tool_trace: list[ToolTraceEvent]
    disclaimer: str
    selective: EvidenceAssessment | None = None

