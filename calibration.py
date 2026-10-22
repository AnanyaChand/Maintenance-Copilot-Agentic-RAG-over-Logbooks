"""Development-only threshold selection, tier-aware empirical uncertainty and audit reports."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .agent import MaintenanceAgent
from .cross_encoder import LocalCrossEncoder, MODEL_NAME, MODEL_PATH
from .evaluation import (EVAL_DIR, citation_counts, percentiles, safety_metrics, sha,
                         validate_suite, write_new)
from .selective import A, B, C, VERSION, FeatureExtractor, SelectiveEngine, Thresholds, decide, make_response
from .tools import DEFAULT_ARTIFACT_DIR, DEFAULT_RECORDS_PATH, DEFAULT_REPORT_PATH, PROJECT_ROOT

SEED = 6062026
GRID = {"reranker": [.85, .97], "fingerprint": [.25, .45], "technical": [.5, 1.],
        "agreement": [0., .5], "corroboration": [0., .5], "evidence_gate": [.55, .70]}


def family_split(cases):
    """Deterministic stratified family split, no retrieval metrics or record labels."""
    strata = {}
    for c in cases:
        key = "supported_family" if not c.expected_refusal else c.category
        strata.setdefault(key, set()).add(c.family)
    development = set()
    for key, families in sorted(strata.items()):
        ordered = sorted(families, key=lambda f: hashlib.sha256(f"{SEED}:{key}:{f}".encode()).hexdigest())
        development.update(ordered[:max(1, int(len(ordered) * .6))])
    result = {"development": [c.id for c in cases if c.family in development],
              "held_out": [c.id for c in cases if c.family not in development]}
    assert not (set(result["development"]) & set(result["held_out"]))
    assert not ({c.family for c in cases if c.id in result["development"]}
                & {c.family for c in cases if c.id in result["held_out"]})
    return result


def threshold_grid():
    return [Thresholds(**dict(zip(GRID, values))) for values in itertools.product(*GRID.values())]


def metrics(rows):
    result = safety_metrics(rows)
    count = len(rows)
    result["coverage"] = sum(r["status"] == "answered" for r in rows) / count if count else None
    accepted = [r for r in rows if r["status"] == "answered"]
    # Query-level selective risk: an unsupported answer OR any citation not labeled relevant.
    result["selective_risk"] = (sum(r["expected_refusal"] or r.get("relevant_pairs", 0) < r.get("citation_pairs", 0)
                                    for r in accepted) / len(accepted)) if accepted else None
    result["tier_counts"] = {tier: sum(r.get("tier") == tier for r in rows) for tier in (A, B, C)}
    return result


def bootstrap_intervals(rows, iterations=500, seed=SEED):
    """Stratified family-cluster bootstrap; query variants are not independent."""
    rng = np.random.default_rng(seed)
    groups = {}
    for row in rows:
        groups.setdefault((row["expected_refusal"], row["family"]), []).append(row)
    strata = [[v for (refusal, _), v in groups.items() if refusal == flag] for flag in (False, True)]
    keys = ("coverage", "supported_answer_rate", "qrel_citation_precision", "citation_precision",
            "false_confidence_rate", "refusal_precision", "refusal_recall", "selective_risk")
    samples = {k: [] for k in keys}
    for _ in range(iterations):
        sample = []
        for groups_in_stratum in strata:
            if groups_in_stratum:
                for index in rng.integers(0, len(groups_in_stratum), len(groups_in_stratum)):
                    sample.extend(groups_in_stratum[int(index)])
        values = metrics(sample)
        for key in keys:
            if values[key] is not None:
                samples[key].append(values[key])
    return {key: np.percentile(values, [2.5, 97.5]).tolist() if values else None
            for key, values in samples.items()}


class EmpiricalUncertainty:
    """Tier-conditional nonconformity percentiles, NOT calibrated error probabilities.

    Fit only on development cases whose selected citations all have positive qrels.
    Tiers with fewer than five samples fall back to the pooled reference. Selection
    and calibration reuse development data; no split-conformal coverage is claimed.
    """
    def __init__(self, references):
        self.references = references

    @classmethod
    def fit(cls, development_rows):
        references = {A: [], B: [], "pooled": []}
        for r in development_rows:
            if (r["status"] == "answered" and not r["expected_refusal"]
                    and r["citation_pairs"] > 0 and r["relevant_pairs"] == r["citation_pairs"]):
                references[r["tier"]].append(r["decision"]["nonconformity"])
                references["pooled"].append(r["decision"]["nonconformity"])
        return cls(references)

    def predict(self, decision):
        values = self.references.get(decision["tier"], [])
        if len(values) < 5:
            values = self.references["pooled"]
        if not values:
            return {"score": 1., "interval": [0., 1.], "samples": 0}
        a = decision["nonconformity"]
        values = np.asarray(values)
        # One minus a conformal-style upper-tail support rank, with +1 smoothing.
        score = float(np.sum(values < a) / (len(values) + 1))
        rng = np.random.default_rng(SEED)
        resampled = rng.choice(values, size=(300, len(values)), replace=True)
        intervals = np.percentile(np.sum(resampled < a, axis=1) / (len(values) + 1), [2.5, 97.5]).tolist()
        return {"score": score, "interval": intervals, "samples": len(values)}


def evaluate_case(case, features, thresholds, record_store, calibration=None):
    decision = decide(features, thresholds)
    uncertainty = calibration.predict(decision) if calibration else {"score": 1., "interval": [0., 1.], "samples": 0}
    # Offline decision replay: real agent/fetch/validation, cached candidate features.
    agent = MaintenanceAgent(search_tool=lambda **kwargs: make_response(features, decision, uncertainty, kwargs["top_k"]),
                             get_record_tool=record_store.__getitem__)
    answer = agent.answer(case.query)
    return answer_row(case, answer, decision)


def answer_row(case, answer, decision):
    statements = answer.observed_issue_patterns + answer.historical_actions + answer.possible_causes
    # Exact source-fidelity accounting is populated by the caller with its record store.
    return {"id": case.id, "family": case.family, "category": case.category,
            "expected_refusal": case.expected_refusal, "status": answer.status,
            "tier": decision["tier"], "decision": decision,
            "citation_pairs": sum(len(s.record_ids) for s in statements),
            "relevant_pairs": sum(rid in case.relevance_grades for s in statements for rid in s.record_ids),
            "claims": len(statements), "answer": answer.model_dump(mode="json")}


def populate_grounding(rows, records):
    from .models import AgentAnswer
    for row in rows:
        row.update(citation_counts(AgentAnswer.model_validate(row["answer"]), records))
    return rows


def feasible(m):
    return ((m["qrel_citation_precision"] or 0.) >= .90
            and m["false_confidence_rate"] == 0. and m["citation_precision"] == 1.)


def sweep(development_cases, features, records):
    table = []
    for index, thresholds in enumerate(threshold_grid()):
        rows = populate_grounding([evaluate_case(c, features[c.id], thresholds, records)
                                   for c in development_cases], records)
        table.append({"index": index, "thresholds": asdict(thresholds), "metrics": metrics(rows)})
    valid = [row for row in table if feasible(row["metrics"])]
    # Fixed objective/tie break established before looking at held-out outcomes.
    pool = valid or table
    chosen = max(pool, key=lambda r: (
        feasible(r["metrics"]), r["metrics"]["supported_answer_rate"] if feasible(r["metrics"]) else 0.,
        r["metrics"]["qrel_citation_precision"] or 0., -(r["metrics"]["false_confidence_rate"] or 0.),
        -r["index"]))
    return table, chosen, bool(valid)


def pareto(table):
    def vector(row):
        m = row["metrics"]
        return (m["supported_answer_rate"] or 0., m["qrel_citation_precision"] or 0.,
                -(m["false_confidence_rate"] if m["false_confidence_rate"] is not None else 1.))
    return [r for r in table if not any(all(a >= b for a, b in zip(vector(other), vector(r)))
            and any(a > b for a, b in zip(vector(other), vector(r))) for other in table)]


def write_curves(directory, table):
    """Dependency-free static SVG plots of every development threshold point."""
    import html
    for metric, title in (("selective_risk", "Selective risk vs coverage"),
                          ("qrel_citation_precision", "Citation relevance vs coverage"),
                          ("false_confidence_rate", "False confidence vs coverage")):
        points = sorted({(r["metrics"]["coverage"], r["metrics"][metric]) for r in table
                         if r["metrics"][metric] is not None})
        circles = "".join(f'<circle cx="{60+x*500:.2f}" cy="{340-y*280:.2f}" r="4" fill="#087e82"/>' for x, y in points)
        # Upper/lower envelope is descriptive, not an interpolated risk guarantee.
        grouped = {}
        for x, y in points:
            grouped.setdefault(x, []).append(y)
        envelope = [(x, max(ys) if metric == "qrel_citation_precision" else min(ys)) for x, ys in sorted(grouped.items())]
        line = " ".join(f"{60+x*500:.2f},{340-y*280:.2f}" for x, y in envelope)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">'
               f'<rect width="640" height="420" fill="white"/><text x="60" y="28" font-size="19">{html.escape(title)}</text>'
               '<path d="M60 60V340H560" fill="none" stroke="#333"/>'
               '<text x="45" y="62">1</text><text x="45" y="345">0</text><text x="555" y="360">1</text>'
               '<text x="245" y="385">Coverage (all development queries)</text>'
               f'<polyline points="{line}" fill="none" stroke="#087e82"/>{circles}'
               '<text x="60" y="410" font-size="11">Development only; dots are threshold combinations; line is empirical envelope.</text></svg>')
        with (directory / f"{metric}.svg").open("x") as handle:
            handle.write(svg)


def run(output: Path, suite_path: Path):
    from .retrieval import HybridSearchEngine
    if output.exists():
        raise FileExistsError("Use a new output directory; calibration artifacts are append-only")
    payload = json.loads(suite_path.read_text())
    if sha(DEFAULT_RECORDS_PATH) != payload["canonical_sha256"]:
        raise ValueError("Canonical dataset changed")
    started = time.perf_counter()
    base = HybridSearchEngine.load(DEFAULT_RECORDS_PATH, DEFAULT_REPORT_PATH, DEFAULT_ARTIFACT_DIR)
    records = base.records_by_id
    cases = validate_suite(payload, records)
    split = family_split(cases)
    output.mkdir(parents=True)
    write_new(output / "split.json", {**split, "seed": SEED, "suite_sha256": sha(suite_path)})
    model = None
    try:
        model = LocalCrossEncoder()
        extractor = FeatureExtractor(base, model)
        startup = (time.perf_counter() - started) * 1000
        features = {}
        # Fit/selection sees development features/labels only. Held-out extraction
        # occurs only AFTER selected thresholds and references are persisted.
        dev = [c for c in cases if c.id in split["development"]]
        held = [c for c in cases if c.id in split["held_out"]]
        for c in dev:
            features[c.id] = extractor.extract(c.query)
            print("development features", c.id, flush=True)
        table, chosen, success = sweep(dev, features, records)
        thresholds = Thresholds(**chosen["thresholds"])
        dev_rows = populate_grounding([evaluate_case(c, features[c.id], thresholds, records) for c in dev], records)
        calibration = EmpiricalUncertainty.fit(dev_rows)
        selection = {"version": VERSION, "suite_sha256": sha(suite_path), "thresholds": asdict(thresholds),
                     "references": calibration.references, "development_ids": split["development"],
                     "development_feasible": success, "canonical_sha256": sha(DEFAULT_RECORDS_PATH),
                     "model_name": MODEL_NAME, "model_sha256": sha(MODEL_PATH / "model.safetensors")}
        write_new(output / "selected.json", selection)
        write_new(output / "sweep.json", {"grid": GRID, "selected_index": chosen["index"],
                                         "rows": table, "pareto": pareto(table)})
        write_curves(output, table)
        live = SelectiveEngine(extractor, thresholds, calibration)
        agent = MaintenanceAgent(search_tool=live.search, get_record_tool=live.get_record)
        held_rows, latencies = [], []
        for c in held:
            start = time.perf_counter()
            answer = agent.answer(c.query)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            assessment = answer.selective
            # Retain the final decision from the response without re-running inference.
            decision = {"tier": assessment.tier, "selected_ids": answer.cited_record_ids,
                        "score": answer.confidence.score, "nonconformity": 1-answer.confidence.score,
                        "exact_match_rescue": assessment.exact_match_rescue,
                        "action_divergence": assessment.action_divergence,
                        "gates": assessment.gates, "limitations": assessment.limitations}
            row = answer_row(c, answer, decision)
            row["latency_ms"] = elapsed
            held_rows.append(row)
            print("held-out", c.id, assessment.tier, flush=True)
        populate_grounding(held_rows, records)
        dev_rows = populate_grounding([evaluate_case(c, features[c.id], thresholds, records, calibration) for c in dev], records)
        comparisons = {}
        for variant, filename in (("baseline", "advanced.baseline.v1.json"), ("old_gated", "advanced.field-gated.v1.json")):
            old = json.loads((EVAL_DIR / "results" / filename).read_text())
            if old["suite_sha256"] != sha(suite_path):
                raise ValueError("Comparator suite differs")
            selected_rows = [dict(r, family=next(c.family for c in cases if c.id == r["id"]))
                             for r in old["cases"] if r["id"] in split["held_out"]]
            comparisons[variant] = {"metrics": metrics(selected_rows),
                                    "bootstrap_95": bootstrap_intervals(selected_rows),
                                    "historical_latency": old["latency"]["hybrid_agent"]}
        observed = metrics(held_rows)
        old_coverage = comparisons["old_gated"]["metrics"]["supported_answer_rate"]
        accepted = success and feasible(observed) and observed["supported_answer_rate"] >= old_coverage + .10
        # A passed experiment remains opt-in; never changes default routing.
        write_new(output / "activation.json", {"ui_eligible": accepted, "default_activation": False,
            "reason": "Requires feasible dev/held-out precision >=.90, zero observed false confidence, and >=10pp supported coverage gain.",
            "selected_sha256": sha(output / "selected.json")})
        n_safety = observed["counts"]["expected_refusals"]
        report = {"version": VERSION, "provisional": True, "human_review_status": "pending",
            "human_validated_metrics": None, "split": split, "selected": selection,
            "development": {"metrics": metrics(dev_rows), "bootstrap_95": bootstrap_intervals(dev_rows)},
            "held_out": {"metrics": observed, "bootstrap_95": bootstrap_intervals(held_rows),
                         "latency": percentiles(latencies),
                         "zero_event_one_sided_95_upper": 1-.05**(1/n_safety) if n_safety and observed["false_confidence_rate"] == 0 else None},
            "comparisons": comparisons, "startup_ms": startup, "ui_eligible": accepted,
            "cases": dev_rows + held_rows, "development_features": features,
            "source_hashes": {str(p.relative_to(PROJECT_ROOT)): sha(p) for p in sorted((PROJECT_ROOT / "src/maintenance_copilot").glob("*.py"))},
            "limitations": ["Previously inspected benchmark; grouped holdout is not genuinely unseen.",
                "Provisional non-exhaustive labels; no operational safety or formal conformal guarantee.",
                "Development set reused for threshold selection and empirical nonconformity reference.",
                "Bootstrap intervals can be degenerate at zero events; inspect the separate binomial upper bound.",
                "Held-out latency is one sequential live pass; comparator latency is historical full-suite timing.",
                "Cross-encoder trained on MS MARCO, not aviation; sigmoid logits are not correctness probabilities."]}
        write_new(output / "report.json", report)
        from .review import export_review
        baseline_rows = json.loads((EVAL_DIR / "results/advanced.baseline.v1.json").read_text())["cases"]
        original_candidates = {row["id"]: row["rankings"]["hybrid"][:5] for row in baseline_rows}
        review_candidates = {c.id: sorted(set(c.expected_relevant_record_ids) | set(original_candidates[c.id])) for c in cases}
        export_review(report, records, output / "review.csv", review_candidates)
        print(json.dumps({"thresholds": asdict(thresholds), "development": report["development"],
                          "held_out": report["held_out"], "ui_eligible": accepted}, indent=2))
    finally:
        if model:
            model.close()
        base.encoder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=EVAL_DIR / "benchmark.v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.suite)


if __name__ == "__main__":
    main()
