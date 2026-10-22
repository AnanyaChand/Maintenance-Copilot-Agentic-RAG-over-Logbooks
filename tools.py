"""Stable tool interfaces used by the later agent layer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

from .models import CanonicalRecord, SearchResponse
from .retrieval import HybridSearchEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS_PATH = PROJECT_ROOT / "data" / "processed" / "records.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "records.report.json"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@lru_cache(maxsize=1)
def _engine() -> HybridSearchEngine:
    from .field_retrieval import with_variant
    variant = os.getenv("RETRIEVAL_VARIANT", "baseline")
    if variant not in {"baseline", "field-aware", "field-gated"}:
        raise ValueError("RETRIEVAL_VARIANT must be baseline, field-aware or field-gated")
    baseline = HybridSearchEngine.load(
        DEFAULT_RECORDS_PATH,
        DEFAULT_REPORT_PATH,
        DEFAULT_ARTIFACT_DIR,
    )
    try:
        return with_variant(baseline, variant)
    except Exception:
        baseline.encoder.close()
        raise


def search_records(
    query: str,
    domain: str = "aviation",
    top_k: int = 5,
) -> SearchResponse:
    """Search MaintNet aviation records using BM25, MiniLM, FAISS, and RRF."""

    return _engine().search(query=query, domain=domain, top_k=top_k)


def get_record(record_id: str) -> CanonicalRecord:
    """Return the full canonical record for an ID produced by search_records."""

    return _engine().get_record(record_id)
