"""Runtime hybrid retrieval with BM25, FAISS cosine search, and RRF."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from .confidence import assess_confidence
from .encoder_process import EncoderProcessError, ProcessSentenceEncoder
from .indexing import (
    BM25_FILENAME,
    DEFAULT_MODEL_NAME,
    FAISS_FILENAME,
    INDEX_VERSION,
    MANIFEST_FILENAME,
    NORMALIZATION_VERSION,
    RECORD_IDS_FILENAME,
    RRF_K,
    RETRIEVAL_POOL_SIZE,
    Encoder,
    bm25_query_tokens,
    load_canonical_records,
    normalized_tokens,
)
from .models import CanonicalRecord, IndexManifest, SearchHit, SearchResponse
from .normalization import DEFAULT_RESOURCES_DIR, NormalizationResources
from .query import normalize_query


LOGGER = logging.getLogger(__name__)
REFUSAL_MESSAGE = "Insufficient evidence in the available maintenance records"


class RetrievalError(RuntimeError):
    """Raised when retrieval artifacts are absent, stale, or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_encoder(model_name: str) -> Encoder:
    try:
        return ProcessSentenceEncoder(model_name)
    except EncoderProcessError as exc:
        raise RetrievalError(
            "Embedding model is not available locally; run `maintenance-copilot build-index` "
            "once with network access"
        ) from exc


class HybridSearchEngine:
    def __init__(
        self,
        *,
        records: list[CanonicalRecord],
        record_ids: list[str],
        bm25: Any,
        vector_index: Any,
        manifest: IndexManifest,
        encoder: Encoder,
        resources: NormalizationResources,
    ) -> None:
        self.records = records
        self.records_by_id = {record.record_id: record for record in records}
        self.record_ids = record_ids
        self.bm25 = bm25
        self.vector_index = vector_index
        self.manifest = manifest
        self.encoder = encoder
        self.resources = resources

    @classmethod
    def load(
        cls,
        records_path: str | Path,
        report_path: str | Path,
        artifact_dir: str | Path,
        *,
        encoder: Encoder | None = None,
        resources: NormalizationResources | None = None,
    ) -> "HybridSearchEngine":
        try:
            import faiss
        except ImportError as exc:
            raise RetrievalError("faiss-cpu is required; install project dependencies") from exc
        # A single FAISS thread is ample for 6,169 vectors and avoids OpenMP
        # runtime conflicts with PyTorch on macOS. It remains user-configurable.
        faiss.omp_set_num_threads(max(1, int(os.getenv("FAISS_NUM_THREADS", "1"))))

        artifact_path = Path(artifact_dir).expanduser().resolve()
        try:
            manifest = IndexManifest.model_validate_json(
                (artifact_path / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            record_ids = json.loads(
                (artifact_path / RECORD_IDS_FILENAME).read_text(encoding="utf-8")
            )
            with (artifact_path / BM25_FILENAME).open("rb") as handle:
                bm25 = pickle.load(handle)
            vector_index = faiss.read_index(str(artifact_path / FAISS_FILENAME))
        except Exception as exc:
            raise RetrievalError(f"Could not load retrieval artifacts from {artifact_path}: {exc}") from exc

        records = load_canonical_records(records_path)
        if manifest.index_version != INDEX_VERSION or manifest.normalization_version != NORMALIZATION_VERSION:
            raise RetrievalError("Index or normalization version is stale; rebuild the index")
        if manifest.record_count != len(records) or len(record_ids) != len(records):
            raise RetrievalError("Index record counts do not match canonical records")
        if record_ids != [record.record_id for record in records]:
            raise RetrievalError("FAISS record mapping does not match canonical record order")
        if int(vector_index.ntotal) != len(records):
            raise RetrievalError("FAISS vector count does not match canonical records")

        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if report.get("source_sha256") != manifest.source_sha256:
            raise RetrievalError("Source hash differs from index manifest; rebuild the index")
        if _sha256(DEFAULT_RESOURCES_DIR / "abbreviations.json") != manifest.abbreviation_sha256:
            raise RetrievalError("Abbreviation dictionary changed; rebuild the index")
        if _sha256(DEFAULT_RESOURCES_DIR / "typo_corrections.json") != manifest.typo_corrections_sha256:
            raise RetrievalError("Typo dictionary changed; rebuild the index")

        active_resources = resources or NormalizationResources.load()
        active_encoder = encoder or _default_encoder(manifest.embedding_model)
        return cls(
            records=records,
            record_ids=record_ids,
            bm25=bm25,
            vector_index=vector_index,
            manifest=manifest,
            encoder=active_encoder,
            resources=active_resources,
        )

    def get_record(self, record_id: str) -> CanonicalRecord:
        try:
            return self.records_by_id[record_id]
        except KeyError as exc:
            raise KeyError(f"Unknown maintenance record ID: {record_id}") from exc

    def _coverage(self, query_tokens: list[str], record: CanonicalRecord) -> float:
        unique_query = set(query_tokens)
        if not unique_query:
            return 0.0
        document = set(
            normalized_tokens(f"{record.problem_normalized} {record.action_normalized}")
        )
        known_idfs = [max(float(value), 0.01) for value in self.bm25.idf.values()]
        default_weight = max(known_idfs, default=1.0)
        weights = {
            token: max(float(self.bm25.idf.get(token, default_weight)), 0.01)
            for token in unique_query
        }
        denominator = sum(weights.values())
        return sum(weight for token, weight in weights.items() if token in document) / denominator

    def search(self, query: str, domain: str = "aviation", top_k: int = 5) -> SearchResponse:
        if domain != "aviation":
            raise ValueError("Only domain='aviation' is supported")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")

        rewrite = normalize_query(query, self.resources)
        bm25_tokens = bm25_query_tokens(query, rewrite.normalized_query)
        bm25_scores = np.asarray(self.bm25.get_scores(bm25_tokens), dtype=np.float64)
        pool_size = min(RETRIEVAL_POOL_SIZE, len(self.records))
        bm25_positions = np.argsort(-bm25_scores, kind="stable")[:pool_size]
        bm25_data = {
            int(position): (rank, float(bm25_scores[position]))
            for rank, position in enumerate(bm25_positions, start=1)
        }

        query_vector = self.encoder.encode(
            [rewrite.normalized_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise RetrievalError(f"Encoder returned invalid query shape {query_vector.shape}")
        norm = float(np.linalg.norm(query_vector[0]))
        if norm > 0:
            query_vector = query_vector / norm
        similarities, positions = self.vector_index.search(query_vector, pool_size)
        vector_data: dict[int, tuple[int, float]] = {}
        for rank, (position, similarity) in enumerate(
            zip(positions[0], similarities[0]), start=1
        ):
            if position >= 0:
                vector_data[int(position)] = (rank, float(similarity))

        rrf_scores: dict[int, float] = {}
        for position, (rank, _) in bm25_data.items():
            rrf_scores[position] = rrf_scores.get(position, 0.0) + 1.0 / (RRF_K + rank)
        for position, (rank, _) in vector_data.items():
            rrf_scores[position] = rrf_scores.get(position, 0.0) + 1.0 / (RRF_K + rank)
        fused = sorted(
            rrf_scores,
            key=lambda position: (-rrf_scores[position], self.record_ids[position]),
        )

        query_tokens = normalized_tokens(rewrite.normalized_query)
        all_hits: list[SearchHit] = []
        for fused_rank, position in enumerate(fused, start=1):
            record = self.records[position]
            bm25_rank, bm25_score = bm25_data.get(position, (None, None))
            vector_rank, vector_similarity = vector_data.get(position, (None, None))
            coverage = self._coverage(query_tokens, record)
            explanations = []
            if bm25_rank is not None:
                explanations.append(
                    f"BM25 rank {bm25_rank} with raw score {bm25_score:.4f}."
                )
            if vector_rank is not None:
                explanations.append(
                    f"Vector rank {vector_rank} with cosine similarity {vector_similarity:.4f}."
                )
            explanations.append(f"IDF-weighted query-token coverage {coverage:.3f}.")
            all_hits.append(
                SearchHit(
                    record_id=record.record_id,
                    problem_original=record.problem_original,
                    action_original=record.action_original,
                    bm25_score=bm25_score,
                    bm25_rank=bm25_rank,
                    vector_similarity=vector_similarity,
                    vector_rank=vector_rank,
                    rrf_score=rrf_scores[position],
                    rrf_rank=fused_rank,
                    query_token_coverage=coverage,
                    match_explanation=explanations,
                )
            )

        visible_hits = all_hits[:top_k]
        # Refusal is a property of retrieval support, not of how many rows the
        # caller asks the UI to display. Always evaluate the first five.
        confidence_hits = all_hits[:5]
        rank_six = all_hits[5].rrf_score if len(all_hits) > 5 else None
        confidence = assess_confidence(confidence_hits, rank_six_rrf_score=rank_six)
        status = "insufficient_evidence" if confidence.should_refuse else "supported"
        response = SearchResponse(
            query=rewrite,
            domain="aviation",
            top_k=top_k,
            status=status,
            refusal_message=REFUSAL_MESSAGE if confidence.should_refuse else None,
            confidence=confidence,
            results=visible_hits,
        )
        LOGGER.info(
            "hybrid_search %s",
            json.dumps(
                {
                    "original_query": query,
                    "normalized_query": rewrite.normalized_query,
                    "status": status,
                    "confidence": confidence.score,
                    "result_ids": [hit.record_id for hit in visible_hits],
                    "scores": [
                        {
                            "record_id": hit.record_id,
                            "bm25": hit.bm25_score,
                            "cosine": hit.vector_similarity,
                            "rrf": hit.rrf_score,
                            "coverage": hit.query_token_coverage,
                        }
                        for hit in visible_hits
                    ],
                },
                separators=(",", ":"),
            ),
        )
        return response
