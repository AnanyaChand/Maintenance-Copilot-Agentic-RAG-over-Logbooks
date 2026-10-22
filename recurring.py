"""Repeated issue patterns in the MaintNet corpus, without aircraft-level claims."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import AgentAnswer, CanonicalRecord
from .tools import DEFAULT_RECORDS_PATH


class RecurringPattern(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    normalized_issue: str
    record_count: int = Field(ge=2)
    record_ids: list[str]
    sample_problems: list[str]
    historical_actions: list[str]


@lru_cache(maxsize=4)
def _load_groups(path: str, modification_time_ns: int) -> dict[str, tuple[CanonicalRecord, ...]]:
    del modification_time_ns  # Cache key invalidates when the processed file changes.
    groups: dict[str, list[CanonicalRecord]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = CanonicalRecord.model_validate_json(line)
                groups[record.problem_normalized].append(record)
    return {key: tuple(value) for key, value in groups.items()}


def detect_recurring_patterns(
    answer: AgentAnswer,
    records_path: str | Path = DEFAULT_RECORDS_PATH,
    *,
    limit: int = 3,
) -> list[RecurringPattern]:
    """Count exact normalized issue repeats for cited records only.

    No aircraft identifier or dates exist in this corpus, so these are dataset
    patterns. A refusal cannot yield a recurring pattern claim.
    """

    if answer.status != "answered" or limit < 1:
        return []
    path = Path(records_path).expanduser().resolve()
    groups = _load_groups(str(path), path.stat().st_mtime_ns)
    cited = set(answer.cited_record_ids)
    seen: set[str] = set()
    patterns: list[RecurringPattern] = []
    for issue, records in groups.items():
        if issue in seen or len(records) < 2 or not any(r.record_id in cited for r in records):
            continue
        seen.add(issue)
        patterns.append(
            RecurringPattern(
                normalized_issue=issue,
                record_count=len(records),
                record_ids=[r.record_id for r in records],
                sample_problems=list(dict.fromkeys(r.problem_original for r in records))[:2],
                historical_actions=list(dict.fromkeys(r.action_original for r in records))[:3],
            )
        )
    patterns.sort(key=lambda p: (-p.record_count, p.record_ids[0]))
    return patterns[:limit]
