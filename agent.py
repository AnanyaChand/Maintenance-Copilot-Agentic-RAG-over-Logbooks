"""Evidence-grounded orchestration over the two maintenance tools."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import time
from uuid import uuid4

from .models import (
    AgentAnswer,
    CanonicalRecord,
    GroundedStatement,
    SearchResponse,
    SynthesisDraft,
    ToolTraceEvent,
)
from .providers import DeterministicEvidenceProvider, SynthesisProvider
from .retrieval import REFUSAL_MESSAGE
from .tools import get_record, search_records


MAINTENANCE_DISCLAIMER = (
    "Historical maintenance records are informational evidence only. Consult approved "
    "maintenance manuals, engineering data, and authorized personnel before performing "
    "maintenance."
)


class GroundingError(ValueError):
    """Raised when a provider draft is not fully supported by fetched records."""


class AgentExecutionError(RuntimeError):
    """Operational failure with the completed and failed tool events attached."""

    def __init__(self, message: str, tool_trace: list[ToolTraceEvent]):
        super().__init__(message)
        self.tool_trace = tool_trace


SearchTool = Callable[[str, str, int], SearchResponse]
GetRecordTool = Callable[[str], CanonicalRecord]


def _contains_verbatim(source: str, excerpt: str) -> bool:
    # Whole fields retain negation and context; substrings can invert meaning.
    return bool(excerpt.strip()) and source == excerpt


def validate_draft(
    draft: SynthesisDraft,
    *,
    searched_ids: set[str],
    fetched_records: dict[str, CanonicalRecord],
) -> None:
    selected = draft.selected_record_ids
    if len(selected) != len(set(selected)):
        raise GroundingError("Provider selected duplicate record IDs")
    if not set(selected).issubset(searched_ids):
        raise GroundingError("Provider cited a record not returned by search_records")
    if set(selected) != set(fetched_records):
        raise GroundingError("Every selected record must be fetched with get_record")
    if not draft.observed_issue_patterns or not draft.historical_actions:
        raise GroundingError("Issue and action evidence are required")
    if draft.possible_causes:
        raise GroundingError("This dataset has no validated cause field; causes must remain empty")
    cited = set()

    groups = (
        ("issue", draft.observed_issue_patterns),
        ("cause", draft.possible_causes),
        ("action", draft.historical_actions),
    )
    for kind, statements in groups:
        for statement in statements:
            if not set(statement.record_ids).issubset(fetched_records):
                raise GroundingError(f"{kind} statement cites an unfetched record")
            for record_id in statement.record_ids:
                cited.add(record_id)
                record = fetched_records[record_id]
                sources = (
                    (record.problem_original,)
                    if kind == "issue"
                    else (record.action_original,)
                    if kind == "action"
                    else (record.problem_original, record.action_original)
                )
                if not any(_contains_verbatim(source, statement.text) for source in sources):
                    raise GroundingError(f"{kind} statement is not complete verbatim evidence for {record_id}")
    if cited != set(selected):
        raise GroundingError("Selected IDs must exactly match statement citations")


def _citation(record_id: str) -> str:
    return f"[Record {record_id}]"


def _render_answer(draft: SynthesisDraft) -> str:
    lines = [
        "The following patterns come only from similar historical maintenance records; "
        "they are not approved repair instructions.",
        "",
        "Observed issue evidence:",
    ]
    for statement in draft.observed_issue_patterns:
        citations = " ".join(_citation(record_id) for record_id in statement.record_ids)
        lines.append(f'- “{statement.text}” {citations}')

    lines.extend(["", "Possible causes:"])
    if draft.possible_causes:
        for statement in draft.possible_causes:
            citations = " ".join(_citation(record_id) for record_id in statement.record_ids)
            lines.append(f'- Explicitly documented: “{statement.text}” {citations}')
    else:
        lines.append("- Causes have not been established by this extractive summary.")

    lines.extend(["", "Historical actions reported:"])
    for statement in draft.historical_actions:
        citations = " ".join(_citation(record_id) for record_id in statement.record_ids)
        lines.append(f'- “{statement.text}” {citations}')
    return "\n".join(lines)


class MaintenanceAgent:
    def __init__(
        self,
        *,
        search_tool: SearchTool = search_records,
        get_record_tool: GetRecordTool = get_record,
        provider: SynthesisProvider | None = None,
        max_evidence_records: int = 3,
    ) -> None:
        if not 1 <= max_evidence_records <= 5:
            raise ValueError("max_evidence_records must be between 1 and 5")
        self.search_tool = search_tool
        self.get_record_tool = get_record_tool
        self.provider = provider or DeterministicEvidenceProvider()
        self.max_evidence_records = max_evidence_records

    def _trace_call(
        self,
        *,
        sequence: int,
        tool_name: str,
        arguments: dict[str, str | int],
        trace: list[ToolTraceEvent],
        call: Callable[[], SearchResponse | CanonicalRecord],
    ) -> tuple[SearchResponse | CanonicalRecord, ToolTraceEvent]:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            result = call()
        except Exception as exc:
            duration = (time.perf_counter() - started) * 1000
            # The event is created for observability before the original error
            # is propagated to the application boundary.
            event = ToolTraceEvent(
                sequence=sequence,
                call_id=str(uuid4()),
                tool_name=tool_name,
                arguments=arguments,
                started_at=started_at,
                duration_ms=duration,
                status="error",
                result_summary="Tool call failed.",
                error=str(exc),
            )
            raise AgentExecutionError(str(exc), [*trace, event]) from exc

        duration = (time.perf_counter() - started) * 1000
        if isinstance(result, SearchResponse):
            result_ids = [hit.record_id for hit in result.results]
            summary = (
                f"Retrieved {len(result.results)} records; status={result.status}; "
                f"confidence={result.confidence.score:.3f}."
            )
        else:
            result_ids = [result.record_id]
            summary = f"Fetched record {result.record_id}."
        event = ToolTraceEvent(
            sequence=sequence,
            call_id=str(uuid4()),
            tool_name=tool_name,
            arguments=arguments,
            started_at=started_at,
            duration_ms=duration,
            status="success",
            result_record_ids=result_ids,
            result_summary=summary,
        )
        return result, event

    def answer(
        self,
        query: str,
        domain: str = "aviation",
        top_k: int = 5,
    ) -> AgentAnswer:
        trace: list[ToolTraceEvent] = []
        search_result, search_event = self._trace_call(
            sequence=1,
            tool_name="search_records",
            arguments={"query": query, "domain": domain, "top_k": top_k},
            trace=trace,
            call=lambda: self.search_tool(query=query, domain=domain, top_k=top_k),
        )
        assert isinstance(search_result, SearchResponse)
        trace.append(search_event)

        return self.answer_from_search(query, search_result, trace)

    def answer_from_search(
        self, query: str, search_result: SearchResponse, trace: list[ToolTraceEvent],
    ) -> AgentAnswer:
        """Continue from an actually executed search (also used by LLM fallback)."""

        if not any(hit.evidence_eligible for hit in search_result.results):
            search_result = search_result.model_copy(update={
                "confidence": search_result.confidence.model_copy(update={
                    "should_refuse": True, "label": "low",
                    "explanation": [*search_result.confidence.explanation,
                                    "FAIL: no displayed candidate is eligible for citation."],
                }),
            })
        if (search_result.confidence.should_refuse
                or search_result.status == "insufficient_evidence"
        ):
            return AgentAnswer(
                query=query,
                normalized_query=search_result.query.normalized_query,
                query_rewrite=search_result.query,
                status="insufficient_evidence",
                answer_text=REFUSAL_MESSAGE,
                cited_record_ids=[],
                evidence=search_result.results,
                confidence=search_result.confidence,
                provider=self.provider.name,
                tool_trace=trace,
                disclaimer=MAINTENANCE_DISCLAIMER,
                selective=search_result.selective,
            )

        selected_hits = [hit for hit in search_result.results if hit.evidence_eligible][: self.max_evidence_records]
        if not selected_hits:
            raise GroundingError("Supported search returned no evidence")
        fetched: dict[str, CanonicalRecord] = {}
        for hit in selected_hits:
            record_result, record_event = self._trace_call(
                sequence=len(trace) + 1,
                tool_name="get_record",
                arguments={"record_id": hit.record_id},
                trace=trace,
                call=lambda record_id=hit.record_id: self.get_record_tool(record_id),
            )
            assert isinstance(record_result, CanonicalRecord)
            if record_result.record_id != hit.record_id:
                raise GroundingError(
                    f"get_record returned {record_result.record_id} for requested {hit.record_id}"
                )
            if (record_result.problem_original != hit.problem_original
                    or record_result.action_original != hit.action_original):
                raise GroundingError("Fetched record differs from search evidence")
            fetched[record_result.record_id] = record_result
            trace.append(record_event)

        draft = self.provider.synthesize(query, list(fetched.values()))
        validate_draft(
            draft,
            searched_ids={hit.record_id for hit in search_result.results},
            fetched_records=fetched,
        )
        cited_ids = draft.selected_record_ids
        answer_text = _render_answer(draft)
        if search_result.selective and search_result.selective.tier == "TIER_B_SUPPORTED_WITH_LIMITATIONS":
            answer_text = ("Historical evidence found with limitations\n\n"
                           + "\n".join(search_result.selective.limitations) + "\n\n" + answer_text)
        return AgentAnswer(
            query=query,
            normalized_query=search_result.query.normalized_query,
            query_rewrite=search_result.query,
            status="answered",
            answer_text=answer_text,
            observed_issue_patterns=draft.observed_issue_patterns,
            possible_causes=draft.possible_causes,
            historical_actions=draft.historical_actions,
            cited_record_ids=cited_ids,
            evidence=[hit for hit in selected_hits if hit.record_id in cited_ids],
            confidence=search_result.confidence,
            provider=self.provider.name,
            tool_trace=trace,
            disclaimer=MAINTENANCE_DISCLAIMER,
            selective=search_result.selective,
        )


def answer_query(
    query: str,
    domain: str = "aviation",
    top_k: int = 5,
) -> AgentAnswer:
    """Run the default evidence-grounded maintenance agent."""

    from .llm_agent import answer_with_mode
    return answer_with_mode(query=query, domain=domain, top_k=top_k)
