"""Opt-in field-aware ranking. No embedding rebuild or baseline artifact writes."""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .indexing import bm25_query_tokens, normalized_tokens, original_tokens
from .retrieval import HybridSearchEngine, REFUSAL_MESSAGE


@dataclass(frozen=True)
class FieldPolicy:
    version: str = "field-ranking-v1"
    problem_original: float = 4.0
    problem_normalized: float = 3.0
    action_original: float = 0.5
    action_normalized: float = 0.5
    technical_exact: float = 6.0
    min_problem_coverage: float = 0.65
    min_relative_score: float = 0.85


POLICY = FieldPolicy()
LOGGER = logging.getLogger(__name__)
ALIASES = {
    "l/h": "left", "lh": "left", "left": "left", "left-hand": "left",
    "r/h": "right", "rh": "right", "right": "right", "right-hand": "right",
    "fwd": "forward", "forward": "forward", "aft": "aft",
}


def technical_tokens(tokens: list[str]) -> set[str]:
    """Exact numbers/part-like identifiers and explicit location aliases only.

    No inferred cylinder numbers, fuzzy identifier matching or ambiguous shorthand
    expansion. Missing requested identifiers cannot be supplied by an action field.
    """
    result = set()
    for token in tokens:
        if token in ALIASES:
            result.add("direction:" + ALIASES[token])
        elif re.fullmatch(r"#?\d+", token):
            result.add("number:" + token.lstrip("#"))
        elif any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            result.add("identifier:" + token)
    return result


class FieldBM25:
    """Independent BM25Okapi fields, query-max-scaled before weighted addition.

    Scaling prevents different field lengths/IDF ranges from silently deciding the
    effective weights. Raw field scores remain available. This is a weighted sum
    of separate BM25 scores, not the BM25F term-frequency formula.
    """
    def __init__(self, records, baseline_idf, policy=POLICY):
        self.policy = policy
        self.idf = baseline_idf  # Keep the existing hybrid confidence coverage unchanged.
        self.records = records
        self.fields = {}
        for field in ("problem_original", "problem_normalized", "action_original", "action_normalized"):
            tokenize = original_tokens if field.endswith("original") else normalized_tokens
            self.fields[field] = BM25Okapi([tokenize(getattr(r, field)) or ["__empty_field__"] for r in records])
        self.technical = [technical_tokens(original_tokens(r.problem_original)) for r in records]
        self.problem_sets = [set(normalized_tokens(r.problem_normalized)) for r in records]
        self.problem_idf = self.fields["problem_normalized"].idf
        self.unknown_idf = max((max(float(v), .01) for v in self.problem_idf.values()), default=1.)

    def components(self, query_tokens):
        original = list(dict.fromkeys(t[5:] for t in query_tokens if t.startswith("orig:")))
        normalized = list(dict.fromkeys(t for t in query_tokens if not t.startswith("orig:")))
        raw, contributions = {}, {}
        for field, index in self.fields.items():
            scores = np.asarray(index.get_scores(original if field.endswith("original") else normalized), dtype=float)
            raw[field] = scores
            positive = np.maximum(scores, 0.)
            peak = float(positive.max(initial=0.))
            contributions[field] = getattr(self.policy, field) * positive / peak if peak else positive
        requested = technical_tokens(original)
        exact = np.array([len(requested & tags) / len(requested) if requested else 0.
                          for tags in self.technical])
        contributions["technical_exact"] = self.policy.technical_exact * exact
        total = sum(contributions.values())
        weights = {t: max(float(self.problem_idf.get(t, self.unknown_idf)), .01) for t in normalized}
        denominator = sum(weights.values())
        coverage = np.array([sum(w for t, w in weights.items() if t in tokens) / denominator
                             if denominator else 0. for tokens in self.problem_sets])
        return total, raw, contributions, requested, coverage

    def get_scores(self, query_tokens):
        return self.components(query_tokens)[0]


class FieldAwareSearchEngine(HybridSearchEngine):
    @classmethod
    def from_baseline(cls, engine, *, gated=False):
        # Own lexical structures; never mutate the loaded baseline engine/index.
        instance = cls(records=engine.records, record_ids=engine.record_ids,
                       bm25=FieldBM25(engine.records, engine.bm25.idf),
                       vector_index=engine.vector_index, manifest=engine.manifest,
                       encoder=engine.encoder, resources=engine.resources)
        instance.gated = gated
        instance.positions = {rid: i for i, rid in enumerate(engine.record_ids)}
        return instance

    def search(self, query, domain="aviation", top_k=5):
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        # Existing RRF and confidence operate on the new lexical ranking, with the
        # same semantic candidates. Gate decisions always inspect five, not top_k.
        response = super().search(query, domain=domain, top_k=50)
        rewrite = response.query
        scores, raw, contributions, requested, coverage = self.bm25.components(
            bm25_query_tokens(query, rewrite.normalized_query))
        hits = response.results
        if self.gated:
            peak = float(scores.max(initial=0.)) or 1.
            rrf_peak = max((h.rrf_score for h in hits), default=1.)
            # Stage two: bounded lexical/technical score + retained semantic RRF.
            rerank = {h.record_id: .7 * float(scores[self.positions[h.record_id]]) / peak
                      + .3 * h.rrf_score / rrf_peak for h in hits}
            hits = sorted(hits, key=lambda h: (-rerank[h.record_id], h.record_id))
        else:
            rerank = {h.record_id: h.rrf_score for h in hits}
        best = rerank[hits[0].record_id] if hits else 1.
        decorated = []
        for rank, hit in enumerate(hits, 1):
            position = self.positions[hit.record_id]
            missing = sorted(requested - self.bm25.technical[position])
            reasons = []
            if self.gated:
                if coverage[position] < POLICY.min_problem_coverage:
                    reasons.append(f"Problem-only coverage below {POLICY.min_problem_coverage:.2f}")
                if missing:
                    reasons.append("Requested technical identifiers absent from problem: " + ", ".join(missing))
                if rerank[hit.record_id] < POLICY.min_relative_score * best:
                    reasons.append(f"Final score below {POLICY.min_relative_score:.2f} of top candidate")
            decorated.append(hit.model_copy(update={
                "field_bm25_scores": {k: float(v[position]) for k, v in raw.items()},
                "field_contributions": {k: float(v[position]) for k, v in contributions.items()},
                "problem_token_coverage": float(coverage[position]),
                "technical_matches": sorted(requested & self.bm25.technical[position]),
                "technical_missing": missing,
                "evidence_eligible": not reasons, "evidence_gate_reasons": reasons,
                "final_score": rerank[hit.record_id], "final_rank": rank,
                "match_explanation": [*hit.match_explanation,
                    f"{POLICY.version}: field-weighted BM25; final rank {rank}.",
                    f"Problem-only coverage {coverage[position]:.3f}; technical missing: {missing}.",
                    *reasons],
            }))
        confidence = response.confidence
        if self.gated and not any(h.evidence_eligible for h in decorated[:5]):
            confidence = confidence.model_copy(update={
                "should_refuse": True, "label": "low",
                "explanation": [*confidence.explanation, "FAIL: no top-five record passed the evidence quality gates."],
            })
        # Do not promote an earlier retrieval refusal after reranking.
        result = response.model_copy(update={
            "top_k": top_k, "results": decorated[:top_k], "confidence": confidence,
            "status": "insufficient_evidence" if confidence.should_refuse else "supported",
            "refusal_message": REFUSAL_MESSAGE if confidence.should_refuse else None,
        })
        LOGGER.info("field_search variant=%s response=%s",
                    "field-gated" if self.gated else "field-aware", result.model_dump_json())
        return result


def with_variant(engine, variant):
    if variant == "baseline":
        return engine
    if variant not in {"field-aware", "field-gated"}:
        raise ValueError("Unknown retrieval variant; use baseline, field-aware or field-gated")
    return FieldAwareSearchEngine.from_baseline(engine, gated=variant == "field-gated")
