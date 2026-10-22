"""Frozen, source-auditable benchmark; never changes production retrieval policy.

Run ``python -m maintenance_copilot.evaluation --help``. No network or LLM calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent import MaintenanceAgent
from .indexing import bm25_query_tokens, load_canonical_records
from .query import normalize_query
from .tools import DEFAULT_ARTIFACT_DIR, DEFAULT_RECORDS_PATH, DEFAULT_REPORT_PATH, PROJECT_ROOT

COUNTS = {"supported": 25, "abbreviation": 10, "noisy": 10, "paraphrase": 10,
          "unsupported": 10, "misleading_overlap": 5, "prompt_injection": 5}
EVAL_DIR = PROJECT_ROOT / "evaluation"


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: Literal["supported", "abbreviation", "noisy", "paraphrase", "unsupported",
                      "misleading_overlap", "prompt_injection"]
    query: str = Field(min_length=1)
    family: str
    expected_relevant_record_ids: list[str]
    relevance_grades: dict[str, int]
    rationale: str = Field(min_length=1)
    expected_evidence_strength: Literal["exact", "strong", "partial"] | None
    expected_refusal: bool

    @model_validator(mode="after")
    def consistent_labels(self):
        ids = self.expected_relevant_record_ids
        if len(ids) != len(set(ids)) or set(ids) != set(self.relevance_grades):
            raise ValueError("Unique IDs must match relevance grades")
        if any(v not in (1, 2, 3) for v in self.relevance_grades.values()):
            raise ValueError("Relevance grades must be 1, 2, or 3")
        supported = self.category in {"supported", "abbreviation", "noisy", "paraphrase"}
        if supported != bool(ids) or supported == self.expected_refusal:
            raise ValueError("Category, evidence and refusal expectation disagree")
        if supported != (self.expected_evidence_strength is not None):
            raise ValueError("Supported cases require evidence strength")
        return self


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    """Results and frozen labels are append-only; use a new version to change labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_suite(payload: dict, records: dict) -> list[Case]:
    cases = [Case.model_validate(row) for row in payload["cases"]]
    if Counter(c.category for c in cases) != COUNTS:
        raise ValueError("Benchmark category counts do not match the required 75 cases")
    if len({c.id for c in cases}) != len(cases) or len({c.query for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs or queries")
    for case in cases:
        for rid in case.expected_relevant_record_ids:
            if rid not in records:
                raise ValueError(f"Unknown record ID {rid}")
    return cases


def freeze(authored: Path, output: Path) -> None:
    """Resolve manually authored families without consulting retrieval rankings.

    Expand qrels only to exact normalized-problem duplicates. Preserve original
    problem/action snapshots for all labeled records for subsequent human review.
    """
    draft = json.loads(authored.read_text())
    records = {r.record_id: r for r in load_canonical_records(DEFAULT_RECORDS_PATH)}
    bases = {row["id"]: row for row in draft["cases"] if "record_ids" in row}
    cases = []
    for row in draft["cases"]:
        base = bases[row["parent"]] if "parent" in row else row
        anchors = base.get("record_ids", [])
        problems = {records[rid].problem_normalized for rid in anchors}
        ids = sorted(rid for rid, r in records.items() if r.problem_normalized in problems)
        strength = base.get("strength")
        grade = {"exact": 3, "strong": 2, "partial": 1}.get(strength, 0)
        cases.append(Case(
            id=row["id"], category=row["category"], query=row["query"],
            family=row.get("parent", row["id"]), expected_relevant_record_ids=ids,
            relevance_grades={rid: grade for rid in ids},
            rationale=base["rationale"] + (
                f" Authored {row['category']} variation of {base['id']}." if "parent" in row else ""),
            expected_evidence_strength=strength, expected_refusal=not ids,
        ).model_dump())
    used = {rid for c in cases for rid in c["expected_relevant_record_ids"]}
    payload = {
        "version": draft["version"], "authorship": draft["authorship"],
        "human_review_status": draft["human_review_status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authored_sha256": sha(authored), "canonical_sha256": sha(DEFAULT_RECORDS_PATH),
        "source_sha256": json.loads(DEFAULT_REPORT_PATH.read_text())["source_sha256"],
        "label_policy": "Manually selected anchors plus exact normalized-problem duplicates only. "
                        "Non-exhaustive qrels: unjudged candidates count as zero, not proven irrelevant. "
                        "Injection and misleading-overlap cases conservatively expect refusal.",
        "cases": cases,
        "evidence_snapshots": {rid: {"problem": records[rid].problem_original,
                                     "action": records[rid].action_original} for rid in sorted(used)},
    }
    validate_suite(payload, records)
    write_new(output, payload)
    print(f"Frozen {len(cases)} cases: {output}\nSHA256 {sha(output)}")


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def ranking_metrics(ranking: list[str], grades: dict[str, int]) -> dict[str, float]:
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate IDs in ranking")
    if not grades:
        raise ValueError("Retrieval metrics require supported qrels")
    hit = {f"Hit@{k}": float(bool(set(ranking[:k]) & grades.keys())) for k in (1, 3, 5)}
    reciprocal = next((1 / i for i, rid in enumerate(ranking[:50], 1) if rid in grades), 0.0)
    dcg = sum((2 ** grades.get(rid, 0) - 1) / math.log2(i + 2)
              for i, rid in enumerate(ranking[:5]))
    ideal = sum((2 ** grade - 1) / math.log2(i + 2)
                for i, grade in enumerate(sorted(grades.values(), reverse=True)[:5]))
    return {**hit, "MRR@50": reciprocal, "nDCG@5": dcg / ideal}


def safety_metrics(rows: list[dict]) -> dict:
    # Errors count as operational failures, never successful refusals.
    expected = sum(r["expected_refusal"] for r in rows)
    refused = sum(r["status"] == "insufficient_evidence" for r in rows)
    correct = sum(r["expected_refusal"] and r["status"] == "insufficient_evidence" for r in rows)
    unsafe = sum(r["expected_refusal"] and r["status"] == "answered" for r in rows)
    pairs = sum(r.get("citation_pairs", 0) for r in rows)
    claims = sum(r.get("claims", 0) for r in rows)
    return {
        "refusal_precision": ratio(correct, refused), "refusal_recall": ratio(correct, expected),
        "false_confidence_rate": ratio(unsafe, expected),
        "citation_precision": ratio(sum(r.get("grounded_pairs", 0) for r in rows), pairs),
        "citation_coverage": ratio(sum(r.get("grounded_claims", 0) for r in rows), claims),
        "qrel_citation_precision": ratio(sum(r.get("relevant_pairs", 0) for r in rows), pairs),
        "supported_answer_rate": ratio(sum(not r["expected_refusal"] and r["status"] == "answered"
                                           for r in rows), len(rows) - expected),
        "counts": {"cases": len(rows), "expected_refusals": expected, "actual_refusals": refused,
                   "correct_refusals": correct, "unsafe_answers": unsafe,
                   "errors": sum(r["status"] == "error" for r in rows),
                   "citation_pairs": pairs, "claims": claims},
    }


def percentiles(values: list[float]) -> dict:
    return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "samples": len(values)} if values else {"p50_ms": None, "p95_ms": None, "samples": 0}


def channel_search(engine, query: str, channel: str) -> list[str]:
    rewrite = normalize_query(query, engine.resources)
    if channel == "bm25":
        scores = engine.bm25.get_scores(bm25_query_tokens(query, rewrite.normalized_query))
        positions = np.argsort(-np.asarray(scores), kind="stable")[:50]
    elif channel == "vector":
        vector = np.asarray(engine.encoder.encode([rewrite.normalized_query], normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        norm = np.linalg.norm(vector[0])
        if norm > 0:
            vector /= norm
        _, positions = engine.vector_index.search(vector, 50)
        positions = positions[0]
    else:
        raise ValueError(channel)
    return [engine.record_ids[int(p)] for p in positions if p >= 0]


def citation_counts(answer, records: dict) -> dict:
    fetched = {e.arguments.get("record_id") for e in answer.tool_trace
               if e.tool_name == "get_record" and e.status == "success"}
    searched = {rid for e in answer.tool_trace if e.tool_name == "search_records" and e.status == "success"
                for rid in e.result_record_ids}
    counts = dict(citation_pairs=0, grounded_pairs=0, claims=0, grounded_claims=0)
    for group, field in ((answer.observed_issue_patterns, "problem_original"),
                         (answer.historical_actions, "action_original"),
                         (answer.possible_causes, None)):
        for statement in group:
            counts["claims"] += 1
            valid = []
            for rid in statement.record_ids:
                counts["citation_pairs"] += 1
                grounded = (rid in fetched and rid in searched and rid in records and field is not None
                            and statement.text == getattr(records[rid], field))
                counts["grounded_pairs"] += int(grounded)
                valid.append(grounded)
            counts["grounded_claims"] += int(bool(valid) and all(valid))
    return counts


def run(suite_path: Path, output: Path, repeats: int, variant: str = "baseline") -> None:
    from .retrieval import HybridSearchEngine
    from .field_retrieval import with_variant, POLICY
    from dataclasses import asdict

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark result: {output}")
    payload = json.loads(suite_path.read_text())
    if payload["canonical_sha256"] != sha(DEFAULT_RECORDS_PATH):
        raise ValueError("Canonical dataset changed since labels were frozen")
    records = {r.record_id: r for r in load_canonical_records(DEFAULT_RECORDS_PATH)}
    cases = validate_suite(payload, records)
    started = time.perf_counter()
    engine = HybridSearchEngine.load(DEFAULT_RECORDS_PATH, DEFAULT_REPORT_PATH, DEFAULT_ARTIFACT_DIR)
    engine = with_variant(engine, variant)
    startup_ms = (time.perf_counter() - started) * 1000
    captured = []

    def search_tool(**kwargs):
        response = engine.search(**kwargs)
        captured.append(response)
        return response

    agent = MaintenanceAgent(search_tool=search_tool, get_record_tool=engine.get_record)
    rows = []
    latencies = {"hybrid_agent": [], "hybrid_search": [], "bm25": [], "vector": []}
    ranks = {channel: [] for channel in ("hybrid", "bm25", "vector")}
    try:
        started = time.perf_counter()
        agent.answer("engine idle override killed engine")
        first_query_ms = (time.perf_counter() - started) * 1000
        for iteration in range(repeats):
            order = list(cases)
            random.Random(20261022 + iteration).shuffle(order)
            for case in order:
                captured.clear()
                row = {"id": case.id, "category": case.category, "query": case.query,
                       "expected_refusal": case.expected_refusal}
                started = time.perf_counter()
                try:
                    answer = agent.answer(case.query)
                    elapsed = (time.perf_counter() - started) * 1000
                    latencies["hybrid_agent"].append(elapsed)
                    search_response = captured[-1]
                    latencies["hybrid_search"].append(answer.tool_trace[0].duration_ms)
                    if iteration == 0:
                        row.update(status=answer.status, confidence=answer.confidence.score,
                                   confidence_explanation=answer.confidence.explanation,
                                   agent_latency_ms=elapsed, citations=answer.cited_record_ids,
                                   tool_trace=[t.model_dump(mode="json") for t in answer.tool_trace],
                                   answer=answer.model_dump(mode="json"), **citation_counts(answer, records))
                        statements = answer.observed_issue_patterns + answer.historical_actions + answer.possible_causes
                        row["relevant_pairs"] = sum(rid in case.relevance_grades for s in statements for rid in s.record_ids)
                        # Separate diagnostic call: no changes to the production top-5 agent workflow.
                        full = engine.search(case.query, top_k=50)
                        assert [h.record_id for h in full.results[:5]] == [h.record_id for h in search_response.results]
                        ranking = [h.record_id for h in full.results]
                        row["rankings"] = {"hybrid": ranking}
                        if not case.expected_refusal:
                            ranks["hybrid"].append({"id": case.id, "category": case.category,
                                                    **ranking_metrics(ranking, case.relevance_grades)})
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    if iteration:
                        raise  # A later instability must not silently disappear in aggregate metrics.
                    if not case.expected_refusal:
                        ranks["hybrid"].append({"id": case.id, "category": case.category,
                                                **ranking_metrics([], case.relevance_grades)})
                for channel in ("bm25", "vector"):
                    started = time.perf_counter()
                    ranking = channel_search(engine, case.query, channel)
                    latencies[channel].append((time.perf_counter() - started) * 1000)
                    if iteration == 0:
                        row.setdefault("rankings", {})[channel] = ranking
                        if not case.expected_refusal:
                            ranks[channel].append({"id": case.id, "category": case.category,
                                                   **ranking_metrics(ranking, case.relevance_grades)})
                if iteration == 0:
                    rows.append(row)
                print(f"repeat={iteration + 1}/{repeats} case={case.id} status={row.get('status', 'measured')}", flush=True)
    finally:
        engine.encoder.close()

    def aggregate(items):
        keys = ("Hit@1", "Hit@3", "Hit@5", "MRR@50", "nDCG@5")
        return {key: float(np.mean([r[key] for r in items])) if items else None for key in keys}

    hardware = {}
    if platform.system() == "Darwin":
        for key in ("hw.model", "hw.memsize", "machdep.cpu.brand_string"):
            try:
                hardware[key] = subprocess.check_output(
                    ["sysctl", "-n", key], text=True, stderr=subprocess.PIPE, timeout=5).strip()
            except (OSError, subprocess.SubprocessError):
                hardware[key] = "unavailable (sandbox or host restriction)"
    result = {
        "variant": variant, "field_policy": asdict(POLICY) if variant != "baseline" else None,
        "created_at": datetime.now(timezone.utc).isoformat(), "suite_version": payload["version"],
        "suite_sha256": sha(suite_path), "human_review_status": payload["human_review_status"],
        "canonical_sha256": sha(DEFAULT_RECORDS_PATH),
        "manifest": engine.manifest.model_dump(), "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "hardware": hardware,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "faiss-cpu", "rank-bm25", "sentence-transformers", "torch", "pydantic")}},
        "source_hashes": {str(p.relative_to(PROJECT_ROOT)): sha(p)
                          for p in sorted((PROJECT_ROOT / "src/maintenance_copilot").glob("*.py"))},
        "repeats": repeats, "startup_ms": startup_ms, "first_query_ms": first_query_ms,
        "latency": {key: percentiles(values) for key, values in latencies.items()},
        "retrieval": {channel: {"overall": aggregate(items),
            "by_category": {cat: aggregate([r for r in items if r["category"] == cat])
                            for cat in ("supported", "abbreviation", "noisy", "paraphrase")},
            "per_case": items} for channel, items in ranks.items()},
        "agent": safety_metrics(rows),
        "safety_by_category": {cat: safety_metrics([r for r in rows if r["category"] == cat])
                               for cat in ("unsupported", "misleading_overlap", "prompt_injection")},
        "cases": sorted(rows, key=lambda r: r["id"]),
        "limitations": ["Assistant-authored labels await human adjudication; not a human-written benchmark.",
            "Non-exhaustive qrels; unjudged matches score zero. Variants share families, not independent trials.",
            "MRR is truncated at 50. Retrieval metrics exclude refusal cases and precede refusal gating.",
            "Channel ablations measure ranking only: no alternative agent/refusal policy is deployed.",
            "Latencies are warm sequential in-process measurements; channel ablations omit hybrid confidence work.",
            "Startup is one cached-model process run, not a download, repeated reliability or peak-memory test.",
            "Injection refusal failures do not by themselves imply execution of malicious instructions.",
            "Baseline model, index artifacts and normalization resources unchanged; variant policy is explicit.",
            "For field-gated, retrieval ranking precedes evidence eligibility filtering; citation metrics reflect filtering."],
    }
    write_new(output, result)
    print(json.dumps({"retrieval": {k: v["overall"] for k, v in result["retrieval"].items()},
                      "agent": result["agent"], "latency": result["latency"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    frozen = sub.add_parser("freeze", help="Freeze source-based labels before any scoring")
    frozen.add_argument("--authored", type=Path, default=EVAL_DIR / "authored_cases.json")
    frozen.add_argument("--output", type=Path, default=EVAL_DIR / "benchmark.v1.json")
    runner = sub.add_parser("run", help="Measure unchanged agent and ranking-only ablations")
    runner.add_argument("--suite", type=Path, default=EVAL_DIR / "benchmark.v1.json")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--repeats", type=int, default=3)
    runner.add_argument("--variant", choices=("baseline", "field-aware", "field-gated"), default="baseline")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.authored, args.output)
    else:
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        run(args.suite, args.output, args.repeats, args.variant)


if __name__ == "__main__":
    main()
