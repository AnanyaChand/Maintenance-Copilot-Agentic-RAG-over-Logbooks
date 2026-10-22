"""Deterministic whole-field claim verification; no LLM grader or entailment claim."""
from __future__ import annotations

import re

from .agent import MAINTENANCE_DISCLAIMER
from .llm_models import CopilotAnswer, LIMITATION, DIVERGENCE, NO_CAUSE

PROCEDURAL = re.compile(r"\b(?:you should|perform|must|replace|approved repair|guaranteed|root cause is)\b", re.I)
CAUSE = re.compile(r"\bdue to\b|\bcaused by\b|^cause\s*:", re.I)
UNCERTAIN = re.compile(r"\b(?:no|not|never|possible|possibly|suspected|suspect|may|might|could|unknown|unable|unconfirmed)\b", re.I)


def observation(rid, source):
    return f"Historical record [{rid}] reports: “{source}”"


def explicit_causes(record):
    return [text for text in (record.problem_original, record.action_original)
            if CAUSE.search(text) and not UNCERTAIN.search(text)]


def limitations_for(records):
    groups = {}
    for record in records:
        groups.setdefault(record.problem_normalized, set()).add(record.action_normalized)
    return [LIMITATION] + ([DIVERGENCE] if any(len(actions) > 1 for actions in groups.values()) else [])


class VerificationFailure(Exception):
    def __init__(self, code, audit):
        self.code, self.audit = code, audit
        super().__init__(code)


def verify_answer(answer: CopilotAnswer, *, query, search, fetched, evidence_trace, known_ids):
    """Reject the entire draft if any claim fails; never leave a misleading fragment."""
    groups = ((answer.summary, "problem"), (answer.similar_historical_findings, "problem"),
              (answer.historical_actions, "action"), (answer.documented_causes, "cause"))
    audit = dict(generated_claims=0, supported_claims=0, rejected_claims=0,
                 generated_citation_pairs=0, valid_id_pairs=0, fetched_id_pairs=0)
    selected = answer.cited_record_ids
    eligible = {h.record_id for h in search.results if h.evidence_eligible}
    for statements, kind in groups:
        for claim in statements:
            audit["generated_claims"] += 1
            audit["generated_citation_pairs"] += len(claim.record_ids)
            audit["valid_id_pairs"] += sum(rid in known_ids for rid in claim.record_ids)
            audit["fetched_id_pairs"] += sum(rid in fetched for rid in claim.record_ids)
            valid = len(claim.record_ids) == 1 and not PROCEDURAL.search(claim.text)
            for rid in claim.record_ids:
                record = fetched.get(rid)
                if record is None or rid not in eligible or rid not in selected:
                    valid = False
                    continue
                sources = ([record.problem_original] if kind == "problem" else
                           [record.action_original] if kind == "action" else explicit_causes(record))
                if claim.text not in [observation(rid, source) for source in sources]:
                    valid = False
            audit["supported_claims" if valid else "rejected_claims"] += 1
    if audit["rejected_claims"]:
        raise VerificationFailure("grounding_failed", audit)
    # Every cited record has exactly one whole problem and action observation.
    if (not selected or len(selected) > 3 or len(set(selected)) != len(selected)
            or not set(selected).issubset(fetched) or not set(selected).issubset(eligible)
            or not answer.summary
            or sorted(s.record_ids[0] for s in answer.similar_historical_findings) != sorted(selected)
            or sorted(s.record_ids[0] for s in answer.historical_actions) != sorted(selected)):
        raise VerificationFailure("grounding_failed", audit)
    records = [fetched[rid] for rid in selected]
    if not answer.documented_causes and any(explicit_causes(r) for r in records):
        raise VerificationFailure("grounding_failed", audit)
    expected_evidence = [{"record_id": r.record_id, "problem_original": r.problem_original,
                          "action_original": r.action_original} for r in records]
    if (answer.query != query or answer.normalized_query != search.query.normalized_query
            or answer.status != "answered" or answer.confidence != search.confidence
            or answer.disclaimer != MAINTENANCE_DISCLAIMER
            or [e.model_dump() for e in answer.evidence] != expected_evidence
            or answer.limitations != limitations_for(records)
            or [e.model_dump(mode="json") for e in answer.tool_trace] != evidence_trace):
        raise VerificationFailure("metadata_mismatch", audit)
    return audit


def render_verified(answer):
    sections = [("Summary", answer.summary), ("Similar historical findings", answer.similar_historical_findings),
                ("Documented causes", answer.documented_causes), ("Historical actions reported", answer.historical_actions)]
    parts = []
    for title, claims in sections:
        parts.append(title)
        parts.extend(claim.text for claim in claims)
        if title == "Documented causes" and not claims:
            parts.append(NO_CAUSE)
        parts.append("")
    parts.extend(answer.limitations)
    return "\n".join(parts)
