"""Provider abstraction with a deterministic, no-key evidence selector."""

from __future__ import annotations

from typing import Protocol
import re

from .models import CanonicalRecord, GroundedStatement, SynthesisDraft


class SynthesisProvider(Protocol):
    name: str

    def synthesize(
        self,
        query: str,
        records: list[CanonicalRecord],
    ) -> SynthesisDraft: ...


_EXPLICIT_CAUSE_RE = re.compile(r"\b(?:due to|caused by|result(?:ed|ing)? from)\b", re.I)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


class DeterministicEvidenceProvider:
    """Selects verbatim evidence and never generates maintenance instructions."""

    name = "deterministic-extractive"

    def synthesize(
        self,
        query: str,
        records: list[CanonicalRecord],
    ) -> SynthesisDraft:
        del query  # Selection is already determined by the validated retriever.
        issues = [
            GroundedStatement(text=record.problem_original, record_ids=[record.record_id])
            for record in records
        ]
        actions = [
            GroundedStatement(text=record.action_original, record_ids=[record.record_id])
            for record in records
        ]
        causes: list[GroundedStatement] = []
        # Cause extraction needs semantic validation; phrase matching alone can
        # mislabel negated or speculative causes. Preserve source fields instead.
        return SynthesisDraft(
            selected_record_ids=[record.record_id for record in records],
            observed_issue_patterns=issues,
            possible_causes=causes,
            historical_actions=actions,
        )
