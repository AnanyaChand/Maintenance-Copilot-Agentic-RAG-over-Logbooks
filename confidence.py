"""Explainable support scoring and refusal policy for hybrid retrieval."""

from __future__ import annotations

from .models import ConfidenceAssessment, SearchHit


MIN_VECTOR_SIMILARITY = 0.45
MIN_QUERY_COVERAGE = 0.30
MIN_CONFIDENCE = 0.58


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def assess_confidence(
    ranked_hits: list[SearchHit],
    *,
    rank_six_rrf_score: float | None,
) -> ConfidenceAssessment:
    if not ranked_hits:
        return ConfidenceAssessment(
            score=0.0,
            label="low",
            should_refuse=True,
            explanation=["No records were retrieved."],
            semantic_signal=0.0,
            lexical_signal=0.0,
            consensus_signal=0.0,
            separation_signal=0.0,
        )

    top = ranked_hits[0]
    cosine = top.vector_similarity if top.vector_similarity is not None else -1.0
    semantic = _clip((cosine - 0.25) / 0.45)
    lexical = top.query_token_coverage
    if top.bm25_rank is not None and top.vector_rank is not None:
        if top.bm25_rank <= 5 and top.vector_rank <= 5:
            consensus = 1.0
        elif top.bm25_rank <= 10 and top.vector_rank <= 10:
            consensus = 0.5
        else:
            consensus = 0.0
    else:
        consensus = 0.0

    if rank_six_rrf_score is None or top.rrf_score <= 0:
        separation = 0.0
    else:
        relative_gap = (top.rrf_score - rank_six_rrf_score) / top.rrf_score
        separation = _clip(relative_gap / 0.35)

    score = _clip(
        0.45 * semantic + 0.30 * lexical + 0.15 * consensus + 0.10 * separation
    )
    gates = {
        "at least two retrieved records": len(ranked_hits) >= 2,
        "top result appears in both retrievers' top 10": (
            top.bm25_rank is not None
            and top.vector_rank is not None
            and top.bm25_rank <= 10
            and top.vector_rank <= 10
        ),
        f"top cosine similarity >= {MIN_VECTOR_SIMILARITY:.2f}": cosine >= MIN_VECTOR_SIMILARITY,
        f"top query-token coverage >= {MIN_QUERY_COVERAGE:.2f}": lexical >= MIN_QUERY_COVERAGE,
        f"combined confidence >= {MIN_CONFIDENCE:.2f}": score >= MIN_CONFIDENCE,
    }
    explanation = [
        f"{'PASS' if passed else 'FAIL'}: {description}" for description, passed in gates.items()
    ]
    should_refuse = not all(gates.values())
    label = "high" if score >= 0.72 and not should_refuse else "medium" if not should_refuse else "low"
    return ConfidenceAssessment(
        score=round(score, 6),
        label=label,
        should_refuse=should_refuse,
        explanation=explanation,
        semantic_signal=round(semantic, 6),
        lexical_signal=round(lexical, 6),
        consensus_signal=round(consensus, 6),
        separation_signal=round(separation, 6),
    )
