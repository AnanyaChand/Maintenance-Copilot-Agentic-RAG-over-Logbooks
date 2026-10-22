from types import SimpleNamespace

import numpy as np
import pytest

from maintenance_copilot.agent import MaintenanceAgent
from maintenance_copilot.field_retrieval import FieldBM25, technical_tokens, with_variant
from maintenance_copilot.indexing import bm25_query_tokens, original_tokens
from maintenance_copilot.retrieval import HybridSearchEngine
from test_retrieval import engine  # shared synthetic encoder/index fixture, no network


def test_exact_technical_aliases_and_identifiers_not_fuzzy():
    assert technical_tokens(original_tokens("LH FWD #2 AB-123")) == {
        "direction:left", "direction:forward", "number:2", "identifier:ab-123"}
    assert technical_tokens(original_tokens("L/H forward 2")) == {
        "direction:left", "direction:forward", "number:2"}
    assert "identifier:ab-124" not in technical_tokens(original_tokens("AB-123"))
    assert technical_tokens(original_tokens("RH aft")) == {"direction:right", "direction:aft"}
    assert not technical_tokens(original_tokens("mag cvr"))


def test_problem_fields_outweigh_action_only_match_and_scores_auditable():
    def record(problem, action):
        return SimpleNamespace(problem_original=problem, problem_normalized=problem,
                               action_original=action, action_normalized=action)
    records = [record("actuator jammed", "inspected"), record("unrelated", "actuator jammed")]
    records += [record(f"filler {i}", "inspected") for i in range(8)]
    index = FieldBM25(records, {})
    tokens = bm25_query_tokens("actuator jammed", "actuator jammed")
    total, raw, contributions, requested, coverage = index.components(tokens)
    assert total[0] == pytest.approx(7.)
    assert total[1] == pytest.approx(1.)
    assert raw["problem_original"][0] > 0
    assert coverage[0] == 1 and coverage[1] == 0
    assert not requested
    np.testing.assert_allclose(total, sum(contributions.values()))
    assert np.isfinite(index.get_scores([])).all()
    assert not index.get_scores([]).any()


def test_baseline_is_not_mutated_and_default_is_identical(engine):
    query = "R/H FWD BAFL SEEAL loose"
    before = engine.search(query).model_dump()
    assert with_variant(engine, "baseline") is engine
    variant = with_variant(engine, "field-aware")
    result = variant.search(query)
    assert variant.vector_index is engine.vector_index
    assert variant.bm25 is not engine.bm25
    assert result.results[0].field_bm25_scores
    assert result.results[0].field_contributions["technical_exact"] == 6.
    assert engine.search(query).model_dump() == before
    with pytest.raises(ValueError):
        with_variant(engine, "typo")


def test_gates_can_only_add_refusals_and_keep_rrf_ranks(engine):
    variant = with_variant(engine, "field-gated")
    for query in ("R/H FWD baffle seal loose", "quantum engine time travel"):
        ungated = HybridSearchEngine.search(variant, query, top_k=50)
        gated = variant.search(query, top_k=50)
        if ungated.confidence.should_refuse:
            assert gated.confidence.should_refuse
        assert {h.record_id: h.rrf_rank for h in ungated.results} == {
            h.record_id: h.rrf_rank for h in gated.results}
        assert [h.final_rank for h in gated.results] == list(range(1, len(gated.results) + 1))
        assert variant.search(query, top_k=1).confidence == gated.confidence


def test_identifier_gate_cannot_be_satisfied_by_action_field(engine):
    variant = with_variant(engine, "field-gated")
    result = variant.search("L/H AFT baffle seal")
    hit = next(h for h in result.results if h.record_id == "T-001")
    assert "direction:left" in hit.technical_missing
    assert "direction:aft" in hit.technical_missing
    assert not hit.evidence_eligible
    assert hit.evidence_gate_reasons


def test_agent_never_fetches_rejected_evidence(engine):
    response = engine.search("R/H FWD BAFL SEEAL loose")
    rejected = response.results[0].model_copy(update={"evidence_eligible": False})
    eligible = response.results[1]
    response = response.model_copy(update={"results": [rejected, eligible]})
    calls = []
    def fetch(record_id):
        calls.append(record_id)
        return engine.get_record(record_id)
    agent = MaintenanceAgent(search_tool=lambda **kw: response, get_record_tool=fetch)
    answer = agent.answer("query")
    assert answer.cited_record_ids == [eligible.record_id]
    assert calls == [eligible.record_id]
    calls.clear()
    response = response.model_copy(update={"results": [rejected]})
    answer = agent.answer("query")
    assert answer.status == "insufficient_evidence"
    assert answer.confidence.should_refuse
    assert not answer.cited_record_ids
    assert not calls
