"""Risk-selective evidence policy, independent of labels and the baseline default."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import logging
import math
import re

from .field_retrieval import FieldAwareSearchEngine, technical_tokens
from .indexing import normalized_tokens, original_tokens
from .models import EvidenceAssessment, SearchResponse
from .retrieval import REFUSAL_MESSAGE

A = "TIER_A_STRONG_EVIDENCE"
B = "TIER_B_SUPPORTED_WITH_LIMITATIONS"
C = "TIER_C_INSUFFICIENT_EVIDENCE"
VARIANTS_WARNING = "Multiple historical action variants found; no single action is endorsed."
VERSION = "selective-v1"
LOGGER = logging.getLogger(__name__)
STOP = set("a an the of to is are was were be been in on at and or for has have had with during after would could not".split())
# Intent constraints, not a list of benchmark case strings or IDs.
AUTHORITY = re.compile(
    r"\b(ignore\b.{0,40}\b(rules|instructions|records)|system\s*:|developer\s*[: ]|"
    r"invent|fabricate|bypass|disable.{0,20}checks|safe to fly|return.to.service|"
    r"certif\w*|approved.{0,25}(repair|procedure)|torque limits|prove.{0,30}cause|"
    r"hide.{0,20}citation)", re.I)


@dataclass(frozen=True)
class Thresholds:
    reranker: float = .97
    fingerprint: float = .45
    technical: float = 1.
    agreement: float = .5
    corroboration: float = 0.
    evidence_gate: float = .70


def fingerprint(text):
    return set(normalized_tokens(text)) - STOP


def conflicts(requested, available):
    conflict = []
    for left, right in (("direction:left", "direction:right"), ("direction:forward", "direction:aft")):
        for wanted, opposite in ((left, right), (right, left)):
            if wanted in requested and wanted not in available and opposite in available:
                conflict.append(wanted + " contradicted by " + opposite)
    for prefix in ("number:", "identifier:"):
        wanted = {t for t in requested if t.startswith(prefix)}
        actual = {t for t in available if t.startswith(prefix)}
        if wanted and actual and not wanted.intersection(actual):
            conflict.append(prefix + " disjoint values")
    return conflict


class FeatureExtractor:
    def __init__(self, engine, cross_encoder):
        self.engine = FieldAwareSearchEngine.from_baseline(engine, gated=True)
        self.cross_encoder = cross_encoder
        self.groups = {}
        for record in engine.records:
            self.groups.setdefault(record.problem_normalized, []).append(record)

    def extract(self, query):
        response = self.engine.search(query, top_k=50)
        hits = response.results[:12]
        normalized = response.query.normalized_query
        logits = self.cross_encoder.predict([
            (normalized, self.engine.get_record(h.record_id).problem_normalized) for h in hits])
        if len(logits) != len(hits) or not all(math.isfinite(float(x)) for x in logits):
            raise ValueError("Invalid cross-encoder output; refusing to synthesize")
        requested = technical_tokens(original_tokens(query))
        qfp = fingerprint(normalized)
        candidates = []
        for hit, logit in zip(hits, logits):
            record = self.engine.get_record(hit.record_id)
            dfp = fingerprint(record.problem_normalized)
            dice = 2 * len(qfp & dfp) / (len(qfp) + len(dfp)) if qfp or dfp else 0.
            similarity = SequenceMatcher(None, normalized, record.problem_normalized).ratio()
            tags = technical_tokens(original_tokens(record.problem_original))
            technical = len(requested & tags) / len(requested) if requested else 1.
            ce = 1 / (1 + math.exp(-max(-30., min(30., float(logit)))))
            overlap = hit.problem_token_coverage or 0.
            group = self.groups[record.problem_normalized]
            actions = {r.action_normalized for r in group if r.action_original.strip()}
            agreement = sum(rank is not None and rank <= 10 for rank in
                            (hit.bm25_rank, hit.vector_rank)) / 2
            score = .45 * ce + .25 * overlap + .20 * dice + .10 * technical
            candidates.append({
                "record_id": hit.record_id, "hit": hit.model_dump(), "cross_logit": float(logit),
                "reranker": ce, "fingerprint": dice, "technical": technical,
                "agreement": agreement, "corroboration": min(1., (len(group) - 1) / 2),
                "evidence_gate": score, "overlap": overlap, "similarity": similarity,
                "conflicts": conflicts(requested, tags), "requested_technical": sorted(requested),
                "action_available": bool(record.action_original.strip()),
                "group": record.problem_normalized, "group_count": len(group),
                "action_variants": len(actions),
            })
        candidates.sort(key=lambda c: (-c["evidence_gate"], -c["cross_logit"], c["record_id"]))
        return {"query": query, "response": response.model_dump(mode="json"),
                "intent_blocked": bool(AUTHORITY.search(query)), "candidates": candidates}


def decide(features, thresholds: Thresholds):
    candidates = features["candidates"]
    gates = []
    if features["intent_blocked"]:
        gates.append("FAIL: request asks for unsupported authority, instructions or evidence manipulation")
    eligible = []
    for candidate in candidates:
        c = candidate
        passed = (not c["conflicts"] and c["action_available"] and c["overlap"] >= .20
                  and all(c[key] >= value for key, value in asdict(thresholds).items()))
        if passed:
            eligible.append(c)
    if features["intent_blocked"] or not eligible:
        return {"tier": C, "selected_ids": [], "score": 0., "nonconformity": 1.,
                "exact_match_rescue": False, "action_divergence": False,
                "limitations": [], "gates": gates + ["FAIL: no admissible problem-grounded evidence"]}
    top = eligible[0]
    # Corroboration never combines unrelated faults. Repeated log entries are not
    # independent confirmations of a cause or a successful repair.
    same = [c for c in eligible if c["group"] == top["group"]][:3]
    conflicting_top = any(c["conflicts"] and c["cross_logit"] >= top["cross_logit"] - 1.
                          for c in candidates[:3])
    rescue = (top["group_count"] == 1 and top["reranker"] >= .98
              and top["overlap"] >= .80 and not conflicting_top and not top["conflicts"]
              and top["action_available"] and top["technical"] == 1.
              and (bool(top["requested_technical"]) or top["similarity"] >= .94))
    strong = (top["reranker"] >= max(.98, thresholds.reranker)
              and top["fingerprint"] >= max(.75, thresholds.fingerprint)
              and top["overlap"] >= .75 and top["technical"] == 1.
              and not conflicting_top and top["action_variants"] <= 1
              and (top["corroboration"] >= .5 or rescue))
    tier = A if strong else B
    limitations = []
    if top["action_variants"] > 1:
        limitations.append(VARIANTS_WARNING)
    if top["group_count"] == 1:
        limitations.append("Sparse evidence: only one record has this normalized fault text.")
    if top["technical"] < 1. or top["fingerprint"] < .75:
        limitations.append("Partial fault fingerprint match; missing details must not be inferred.")
    if tier == B:
        limitations.append("Historical descriptions only; no cause, approved procedure or action recommendation is established.")
    gates += [f"PASS: {key} {top[key]:.3f} >= {value:.3f}" for key, value in asdict(thresholds).items()]
    gates += ["PASS: problem evidence, action availability and critical contradiction checks"]
    gates += ["WARNING: " + limitation for limitation in limitations]
    return {"tier": tier, "selected_ids": [c["record_id"] for c in same],
            "score": top["evidence_gate"], "nonconformity": 1 - top["evidence_gate"],
            "exact_match_rescue": bool(rescue and tier == A),
            "action_divergence": top["action_variants"] > 1,
            "limitations": limitations, "gates": gates}


def make_response(features, decision, uncertainty, top_k=5):
    response = SearchResponse.model_validate(features["response"])
    by_id = {h.record_id: h for h in response.results}
    selected = decision["selected_ids"][:min(3, top_k)]
    # No low-confidence neighbors are emitted as evidence, including on refusal.
    hits = [by_id[rid].model_copy(update={"evidence_eligible": True}) for rid in selected]
    assessment = EvidenceAssessment(
        tier=decision["tier"], uncertainty=uncertainty["score"],
        uncertainty_interval=uncertainty["interval"], calibration_samples=uncertainty["samples"],
        gates=decision["gates"], limitations=decision["limitations"],
        action_divergence=decision["action_divergence"], exact_match_rescue=decision["exact_match_rescue"])
    refused = decision["tier"] == C
    return response.model_copy(update={
        "top_k": top_k, "results": hits, "selective": assessment,
        "status": "insufficient_evidence" if refused else "supported",
        "refusal_message": REFUSAL_MESSAGE if refused else None,
        "confidence": response.confidence.model_copy(update={
            "score": decision["score"], "label": "low" if refused else "high" if decision["tier"] == A else "medium",
            "should_refuse": refused, "explanation": decision["gates"],
        }),
    })


class SelectiveEngine:
    def __init__(self, extractor, thresholds, calibration):
        self.extractor, self.thresholds, self.calibration = extractor, thresholds, calibration

    def get_record(self, record_id):
        return self.extractor.engine.get_record(record_id)

    def search(self, query, domain="aviation", top_k=5):
        if domain != "aviation" or not 1 <= top_k <= 50:
            raise ValueError("Only aviation domain and top_k 1..50 are supported")
        features = self.extractor.extract(query)
        decision = decide(features, self.thresholds)
        uncertainty = self.calibration.predict(decision)
        LOGGER.info("selective_decision query=%r decision=%s uncertainty=%s", query, decision, uncertainty)
        return make_response(features, decision, uncertainty, top_k)
