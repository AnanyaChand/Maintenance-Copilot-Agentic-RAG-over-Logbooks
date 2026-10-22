"""Persistent BM25 and FAISS index construction."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
import re
from typing import Any, Protocol, Sequence
import unicodedata

# FAISS and PyTorch both use OpenMP on macOS. Fixing the small-corpus runtime
# to one thread before either library loads prevents duplicate runtime startup.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from .models import CanonicalRecord, IndexManifest, IngestionReport
from .normalization import DEFAULT_RESOURCES_DIR


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_VERSION = "hybrid-rrf-v1"
NORMALIZATION_VERSION = "dictionary-normalization-v1"
RRF_K = 60
RETRIEVAL_POOL_SIZE = 50

BM25_FILENAME = "bm25.pkl"
FAISS_FILENAME = "faiss.index"
EMBEDDINGS_FILENAME = "embeddings.npy"
RECORD_IDS_FILENAME = "record_ids.json"
MANIFEST_FILENAME = "manifest.json"

_NORMALIZED_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_ORIGINAL_TOKEN_RE = re.compile(r"#[0-9]+|[a-z0-9]+(?:[/.-][a-z0-9]+)+|[a-z0-9]+")


class Encoder(Protocol):
    def encode(self, sentences: str | Sequence[str], **kwargs: Any) -> np.ndarray: ...


class IndexBuildError(RuntimeError):
    """Raised when persistent retrieval artifacts cannot be built safely."""


def normalized_tokens(text: str) -> list[str]:
    return _NORMALIZED_TOKEN_RE.findall(text.lower())


def original_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _ORIGINAL_TOKEN_RE.findall(normalized)


def bm25_document_tokens(record: CanonicalRecord) -> list[str]:
    normalized = normalized_tokens(
        f"{record.problem_normalized} {record.action_normalized}"
    )
    exact = original_tokens(f"{record.problem_original} {record.action_original}")
    return normalized + normalized + [f"orig:{token}" for token in exact]


def bm25_query_tokens(original_query: str, normalized_query: str) -> list[str]:
    normalized = normalized_tokens(normalized_query)
    exact = original_tokens(original_query)
    return normalized + normalized + [f"orig:{token}" for token in exact]


def embedding_text(record: CanonicalRecord) -> str:
    return f"{record.problem_normalized} {record.action_normalized}".strip()


def load_canonical_records(path: str | Path) -> list[CanonicalRecord]:
    records_path = Path(path).expanduser().resolve()
    if not records_path.is_file():
        raise IndexBuildError(f"Canonical record file does not exist: {records_path}")
    records: list[CanonicalRecord] = []
    seen_ids: set[str] = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = CanonicalRecord.model_validate_json(line)
            except Exception as exc:
                raise IndexBuildError(
                    f"Invalid canonical record on JSONL line {line_number}: {exc}"
                ) from exc
            if record.record_id in seen_ids:
                raise IndexBuildError(f"Duplicate canonical record ID: {record.record_id}")
            seen_ids.add(record.record_id)
            records.append(record)
    if not records:
        raise IndexBuildError("Canonical record file contains no records")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_default_encoder(model_name: str) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise IndexBuildError(
            "sentence-transformers is required; install the project dependencies"
        ) from exc
    return SentenceTransformer(model_name)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise IndexBuildError(f"Encoder returned shape {vectors.shape}; expected 2 dimensions")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise IndexBuildError("Encoder produced a zero-length document embedding")
    return vectors / norms


def build_hybrid_index(
    records_path: str | Path,
    report_path: str | Path,
    artifact_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Encoder | None = None,
    show_progress: bool = True,
) -> IndexManifest:
    """Build and persist exact BM25 and cosine-similarity retrieval artifacts."""

    try:
        from rank_bm25 import BM25Okapi
        import faiss
    except ImportError as exc:
        raise IndexBuildError(
            "rank-bm25 and faiss-cpu are required; install the project dependencies"
        ) from exc
    faiss.omp_set_num_threads(max(1, int(os.getenv("FAISS_NUM_THREADS", "1"))))

    records = load_canonical_records(records_path)
    report_file = Path(report_path).expanduser().resolve()
    try:
        ingestion_report = IngestionReport.model_validate_json(
            report_file.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise IndexBuildError(f"Invalid ingestion report {report_file}: {exc}") from exc
    if ingestion_report.loaded_records != len(records):
        raise IndexBuildError(
            "Ingestion report count does not match canonical record count: "
            f"{ingestion_report.loaded_records} != {len(records)}"
        )

    tokenized_corpus = [bm25_document_tokens(record) for record in records]
    bm25 = BM25Okapi(tokenized_corpus)

    active_encoder = encoder or _load_default_encoder(model_name)
    encoded = active_encoder.encode(
        [embedding_text(record) for record in records],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    embeddings = _normalize_rows(encoded)
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)

    output = Path(artifact_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / BM25_FILENAME).open("wb") as handle:
        pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
    faiss.write_index(index, str(output / FAISS_FILENAME))
    np.save(output / EMBEDDINGS_FILENAME, embeddings, allow_pickle=False)
    (output / RECORD_IDS_FILENAME).write_text(
        json.dumps([record.record_id for record in records], indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = IndexManifest(
        index_version=INDEX_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        embedding_model=model_name,
        source_sha256=ingestion_report.source_sha256,
        record_count=len(records),
        embedding_dimensions=int(embeddings.shape[1]),
        rrf_k=RRF_K,
        bm25_pool_size=RETRIEVAL_POOL_SIZE,
        vector_pool_size=RETRIEVAL_POOL_SIZE,
        abbreviation_sha256=_sha256(DEFAULT_RESOURCES_DIR / "abbreviations.json"),
        typo_corrections_sha256=_sha256(DEFAULT_RESOURCES_DIR / "typo_corrections.json"),
    )
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
