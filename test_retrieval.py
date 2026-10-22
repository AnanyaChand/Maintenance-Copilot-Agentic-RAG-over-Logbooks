from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

from maintenance_copilot.indexing import build_hybrid_index
from maintenance_copilot.ingestion import ingest_csv
from maintenance_copilot.retrieval import HybridSearchEngine, REFUSAL_MESSAGE


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_maintnet.csv"


class KeywordEncoder:
    keywords = ("baffle", "seal", "cylinder", "gasket", "engine", "battery")

    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        vectors = []
        for sentence in sentences:
            lowered = sentence.lower()
            vector = np.asarray(
                [float(lowered.count(keyword)) for keyword in self.keywords],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            if norm:
                vector /= norm
            vectors.append(vector)
        return np.vstack(vectors)


@pytest.fixture()
def engine(tmp_path):
    ingestion = ingest_csv(FIXTURE)
    records_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "records.report.json"
    records_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in ingestion.records),
        encoding="utf-8",
    )
    report_path.write_text(ingestion.report.model_dump_json(indent=2), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    encoder = KeywordEncoder()
    build_hybrid_index(
        records_path,
        report_path,
        artifacts,
        model_name="test/keyword-encoder",
        encoder=encoder,
        show_progress=False,
    )
    return HybridSearchEngine.load(
        records_path,
        report_path,
        artifacts,
        encoder=encoder,
    )


def test_hybrid_search_exposes_scores_ranks_and_support(engine):
    response = engine.search("R/H FWD BAFL SEEAL loose", top_k=2)

    assert response.status == "supported"
    assert response.refusal_message is None
    assert response.results[0].record_id == "T-001"
    assert response.results[0].bm25_rank == 1
    assert response.results[0].vector_rank == 1
    assert response.results[0].bm25_score is not None
    assert response.results[0].vector_similarity > 0.9
    assert response.results[0].rrf_score > 0
    assert response.results[0].query_token_coverage == pytest.approx(1.0)
    assert response.results[0].match_explanation


def test_unsupported_query_returns_exact_refusal(engine):
    response = engine.search("galley coffee maker display", top_k=2)

    assert response.status == "insufficient_evidence"
    assert response.refusal_message == REFUSAL_MESSAGE
    assert response.confidence.should_refuse is True


def test_get_record_uses_record_id(engine):
    record = engine.get_record("T-002")
    assert record.problem_original == "#2 CYL GSK leaking"


def test_non_aviation_domain_is_rejected(engine):
    with pytest.raises(ValueError, match="Only domain='aviation'"):
        engine.search("engine leak", domain="automotive")


def test_confidence_does_not_depend_on_display_top_k(engine):
    response = engine.search("R/H FWD BAFL SEEAL loose", top_k=1)

    assert len(response.results) == 1
    assert response.status == "supported"
