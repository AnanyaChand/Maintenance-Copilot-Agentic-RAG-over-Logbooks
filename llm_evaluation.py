"""Separate experimental comparison. --provider configured explicitly makes API calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .agent import MaintenanceAgent
from .evaluation import validate_suite, sha, write_new, percentiles
from .llm_agent import answer_with_mode, PROMPT_VERSION
from .llm_config import LLMConfig
from .llm_mock import MockToolCallingProvider
from .retrieval import HybridSearchEngine
from .tools import DEFAULT_RECORDS_PATH, DEFAULT_REPORT_PATH, DEFAULT_ARTIFACT_DIR, PROJECT_ROOT


def fraction(n, d):
    return {"value": n / d if d else None, "numerator": n, "denominator": d}


def delivered_metrics(answer, search):
    """Coverage of selected source fields, not diagnosis quality or human relevance."""
    findings = {rid for s in answer.observed_issue_patterns for rid in s.record_ids}
    actions = {rid for s in answer.historical_actions for rid in s.record_ids}
    citations = [rid for s in (answer.observed_issue_patterns + answer.historical_actions + answer.possible_causes)
                 for rid in s.record_ids]
    returned = {h.record_id for h in search.results}
    fetched = {rid for event in answer.tool_trace if event.tool_name == "get_record" and event.status == "success"
               for rid in event.result_record_ids}
    capacity = min(3, sum(h.evidence_eligible for h in search.results)) if answer.status == "answered" else 0
    return dict(citation_pairs=len(citations), valid_id_pairs=sum(rid in returned for rid in citations),
                fetched_id_pairs=sum(rid in fetched for rid in citations), complete_records=len(findings & actions & set(answer.cited_record_ids)),
                available_record_slots=capacity)


def aggregate(rows, path):
    stats = [r[path] for r in rows]
    total = lambda key: sum(s.get(key, 0) for s in stats)
    return {"valid_citation_id_rate": fraction(total("valid_id_pairs"), total("citation_pairs")),
            "fetched_evidence_citation_rate": fraction(total("fetched_id_pairs"), total("citation_pairs")),
            "answer_completeness": fraction(total("complete_records"), total("available_record_slots")),
            "latency": percentiles([s["latency_ms"] for s in stats])}


def run(output: Path, provider_kind="mock"):
    if output.exists():
        raise ValueError("Choose a new report filename; existing artifacts are preserved")
    config = (LLMConfig(requested=True, enabled=True, model="mock-protocol", timeout=30)
              if provider_kind == "mock" else LLMConfig.from_env())
    if not config.enabled:
        raise ValueError("Configured evaluation requires valid, explicitly enabled LLM settings")
    suite_path = PROJECT_ROOT / "evaluation" / "benchmark.v1.json"
    suite = json.loads(suite_path.read_text())
    if sha(DEFAULT_RECORDS_PATH) != suite["canonical_sha256"]:
        raise ValueError("Canonical data does not match frozen suite")
    startup = time.perf_counter()
    engine = HybridSearchEngine.load(DEFAULT_RECORDS_PATH, DEFAULT_REPORT_PATH, DEFAULT_ARTIFACT_DIR)
    startup_ms = (time.perf_counter() - startup) * 1000
    rows = []
    try:
        cases = validate_suite(suite, engine.records_by_id)
        captured = []
        def search(**kwargs):
            result = engine.search(**kwargs)
            captured.append(result)
            return result
        agent = MaintenanceAgent(search_tool=search, get_record_tool=engine.get_record)
        for case in cases:
            start = time.perf_counter()
            baseline = agent.answer(case.query)
            baseline_ms = (time.perf_counter() - start) * 1000
            baseline_search = captured[-1]
            start = time.perf_counter()
            candidate = answer_with_mode(case.query, agent=agent, config=config,
                provider=MockToolCallingProvider() if provider_kind == "mock" else None)
            candidate_ms = (time.perf_counter() - start) * 1000
            mode = getattr(candidate, "mode_info", None)
            refusal_preserved = (candidate.status == baseline.status and candidate.answer_text == baseline.answer_text
                                 and len(candidate.tool_trace) == 1 and (not mode or not mode.llm_used))
            rows.append(dict(id=case.id, expected_refusal_provisional=case.expected_refusal,
                deterministic={**delivered_metrics(baseline, baseline_search), "latency_ms": baseline_ms,
                               "status": baseline.status, "cited_record_ids": baseline.cited_record_ids},
                candidate={**delivered_metrics(candidate, baseline_search), "latency_ms": candidate_ms,
                           "status": candidate.status, "answer": candidate.model_dump(mode="json")},
                baseline_refused=baseline.status == "insufficient_evidence", refusal_preserved=refusal_preserved,
                generation_audit=mode.model_dump() if mode else None))
            print(f"case={case.id} mode={mode.mode if mode else 'retrieval_refusal'}", flush=True)
    finally:
        engine.encoder.close()
    audits = [r["generation_audit"] for r in rows if r["generation_audit"]]
    total = lambda key: sum(a[key] for a in audits)
    refusals = [r for r in rows if r["baseline_refused"]]
    result = dict(
        status="provisional / not human-validated", provider_kind=provider_kind,
        actual_model=config.model, mock_only=provider_kind == "mock", prompt_version=PROMPT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(), suite_sha256=sha(suite_path),
        canonical_sha256=sha(DEFAULT_RECORDS_PATH), startup_ms=startup_ms,
        source_hashes={str(p.relative_to(PROJECT_ROOT)): sha(p) for p in sorted(
            [*Path(__file__).parent.glob("llm_*.py"), Path(__file__).parent / "agent.py",
             Path(__file__).parent / "models.py", Path(__file__).parent / "resources" / "llm_system_v1.txt"])},
        deterministic=aggregate(rows, "deterministic"), candidate_delivered=aggregate(rows, "candidate"),
        generation={"valid_citation_id_rate": fraction(total("valid_id_pairs"), total("generated_citation_pairs")),
            "fetched_evidence_citation_rate": fraction(total("fetched_id_pairs"), total("generated_citation_pairs")),
            "claim_support_precision": fraction(total("supported_claims"), total("generated_claims")),
            "unsupported_claim_rejection_count": total("rejected_claims"),
            "fallback_count": sum(a["fallback"] for a in audits),
            "schema_rejection_count": sum(any(e["tool_name"] == "llm_fallback" and
                e["arguments"].get("reason_code") == "malformed_output" for e in r["candidate"]["answer"]["tool_trace"]) for r in rows),
            "provider_requests": total("provider_requests"),
            "accepted_llm_answers": sum(a["mode"] == "llm" for a in audits)},
        refusal_preservation=fraction(sum(r["refusal_preserved"] for r in refusals), len(refusals)),
        limitations=["Mock mode tests protocol execution; it measures no frontier model quality or API latency.",
            "Baseline retrieval refusals are preserved, including over-refusals; baseline safety failures may persist.",
            "Valid citation ID means present in actual search results; fetched evidence is separately checked.",
            "Claim precision measures schema-valid generated statements under a whole-field verifier, not human semantic relevance.",
            "Malformed output has no countable claims and remains visible as fallback, never an implicit success.",
            "Completeness is records with both issue and action / up to three eligible slots; it is not factual coverage of a diagnosis.",
            "One sequential pass; timings exclude engine startup and UI rendering. First inference may warm model caches.",
            "No operational safety or model improvement is established; human review is pending."], cases=rows)
    # Baseline emitted claims are validated by the existing whole-field validator.
    pairs = sum(r["deterministic"]["citation_pairs"] for r in rows)
    result["deterministic"]["claim_support_precision"] = fraction(pairs, pairs)
    write_new(output, result)
    print(f"Saved {provider_kind} experiment: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock", "configured"), default="mock")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.provider)


if __name__ == "__main__":
    main()
