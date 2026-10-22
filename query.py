"""Auditable, conservative query rewriting."""

from __future__ import annotations

from .models import QueryRewrite, RewriteChange
from .normalization import NormalizationResources, normalize_text_with_trace


def normalize_query(
    query: str, resources: NormalizationResources | None = None
) -> QueryRewrite:
    if not query or not query.strip():
        raise ValueError("query must not be blank")
    result = normalize_text_with_trace(query, resources)
    if not result.text:
        raise ValueError("query has no searchable tokens after normalization")
    return QueryRewrite(
        original_query=query,
        normalized_query=result.text,
        expansions_applied=[
            RewriteChange(
                original=item.original,
                replacement=item.replacement,
                occurrences=item.occurrences,
            )
            for item in result.expansions
        ],
        typo_corrections_applied=[
            RewriteChange(
                original=item.original,
                replacement=item.replacement,
                occurrences=item.occurrences,
            )
            for item in result.typo_corrections
        ],
    )
