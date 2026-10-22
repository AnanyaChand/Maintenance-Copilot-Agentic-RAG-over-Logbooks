"""Maintenance Copilot ingestion and normalization package."""

import os

# Must be set before Pydantic or NumPy can transitively initialize a numerical
# runtime. One thread is sufficient for this 6,169-record local corpus.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .models import CanonicalRecord, IngestionReport, SearchResponse
from .query import normalize_query

__all__ = ["CanonicalRecord", "IngestionReport", "SearchResponse", "normalize_query"]

