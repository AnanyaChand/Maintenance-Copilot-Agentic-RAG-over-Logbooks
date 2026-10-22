"""Auditable curated extraction from PROBLEM only. No action or outcome inference."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field

from .normalization import normalize_text_with_trace

FINGERPRINT_VERSION = "maintnet-problem-fingerprint-v1"
RESOURCE_PATH = Path(__file__).with_name("resources") / "pattern_terms.v1.json"


class Fingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: str
    components: list[str]
    subsystems: list[str]
    symptoms: list[str]
    states: list[str]
    locations: list[str]
    technical_identifiers: list[str]
    original_technical_tokens: list[str]
    abbreviation_expanded_terms: list[str]
    negated_symptoms: list[str]
    extraction_audit: list[dict]
    version: str = FINGERPRINT_VERSION
    resource_checksum: str


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=8)
def _resource(checksum):
    resource = json.loads(RESOURCE_PATH.read_text())
    compiled = {}
    for category in ("components", "subsystems", "symptoms", "states", "locations"):
        compiled[category] = [(term, re.compile(r"(?<![a-z0-9])(?:" + "|".join(
            re.escape(a) for a in sorted(aliases, key=lambda a: (-len(a), a))) + r")(?![a-z0-9])", re.I))
            for term, aliases in resource[category].items()]
    return resource, compiled


def fingerprint(record):
    checksum = digest(RESOURCE_PATH)
    resource, compiled = _resource(checksum)
    audit, extracted, negated = [], {}, set()
    # Existing canonical normalization stays unchanged. Raw tokens are audited too.
    text = record.problem_normalized
    for category, patterns in compiled.items():
        found = set()
        for term, pattern in patterns:
            for match in pattern.finditer(text):
                found.add(term)
                audit.append(dict(category=category, term=term, matched=match.group(),
                                  source="problem_normalized", start=match.start(), end=match.end()))
                if category == "symptoms" and re.search(r"\b(?:no|not|without)\s+(?:\w+\s+){0,2}$", text[:match.start()]):
                    negated.add(term)
        extracted[category] = sorted(found)
    identifiers, raw_tokens = set(), set()
    # Preserve punctuation-bearing alphanumeric identifiers and numbered components.
    for match in re.finditer(r"(?<![A-Za-z0-9])(?:[A-Za-z]+[A-Za-z0-9]*[-/][A-Za-z0-9-]*\d[A-Za-z0-9-]*|[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+)(?![A-Za-z0-9])", record.problem_original):
        token = match.group()
        raw_tokens.add(token)
        identifiers.add("token:" + token.upper())
        audit.append(dict(category="technical_identifiers", term="token:" + token.upper(),
                          matched=token, source="problem_original", start=match.start(), end=match.end()))
    for match in re.finditer(r"#\s*(\d+)\b|\b(?:number|no\.?|cyl(?:inder)?)\s*#?\s*(\d+)\b", record.problem_original, re.I):
        number = match.group(1) or match.group(2)
        raw_tokens.add(match.group())
        identifiers.add("number:" + number)
        audit.append(dict(category="technical_identifiers", term="number:" + number,
                          matched=match.group(), source="problem_original", start=match.start(), end=match.end()))
    # A numeric conflict such as 300 vs 400 RPM is also significant; numbers never
    # come from ACTION text. Include all explicit problem numerals conservatively.
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", record.problem_original))
    identifiers.update("numeric:" + n for n in numbers)
    raw_tokens.update(re.findall(r"\b\d+(?:\.\d+)?\b", record.problem_original))
    for number in sorted(numbers):
        audit.append(dict(category="technical_identifiers", term="numeric:" + number,
                          matched=number, source="problem_original", rule="explicit numeral"))
    rewrite = normalize_text_with_trace(record.problem_original)
    for change in rewrite.expansions:
        audit.append(dict(category="abbreviation_expanded_terms", term=change.replacement,
                          matched=change.original, source="problem_original", occurrences=change.occurrences))
    return Fingerprint(record_id=record.record_id, **extracted,
        technical_identifiers=sorted(identifiers), original_technical_tokens=sorted(raw_tokens),
        abbreviation_expanded_terms=sorted({c.replacement for c in rewrite.expansions}),
        negated_symptoms=sorted(negated), extraction_audit=audit, resource_checksum=checksum)


def meaningful(fp):
    return set(fp.components) | set(fp.symptoms) | set(fp.technical_identifiers)


def conflicts(a, b):
    resource, _ = _resource(a.resource_checksum)
    reasons = []
    for axis, terms in resource["location_axes"].items():
        left, right = set(a.locations) & set(terms), set(b.locations) & set(terms)
        if len(left) > 1 or len(right) > 1 or (left and right and left != right):
            reasons.append("location:" + axis)
    for kind in ("number:", "numeric:", "token:"):
        left = {t for t in a.technical_identifiers if t.startswith(kind)}
        right = {t for t in b.technical_identifiers if t.startswith(kind)}
        if left and right and left != right:
            reasons.append("technical:" + kind.rstrip(":"))
    for symptom in set(a.symptoms) & set(b.symptoms):
        if (symptom in a.negated_symptoms) != (symptom in b.negated_symptoms):
            reasons.append("negation:" + symptom)
    # Explicit opposing states should not be merged even with similar embeddings.
    for states in ({"open", "closed"}, {"high", "low"}):
        left, right = set(a.states) & states, set(b.states) & states
        if len(left) > 1 or len(right) > 1 or (left and right and left != right):
            reasons.append("state:" + "/".join(sorted(states)))
    return reasons
