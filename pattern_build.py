"""Append-only, offline build of historical pattern artifacts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import faiss
from huggingface_hub import try_to_load_from_cache
import numpy as np

from .encoder_process import ProcessSentenceEncoder
from .models import CanonicalRecord
from .pattern_fingerprints import digest, RESOURCE_PATH, FINGERPRINT_VERSION
from .pattern_graph import analyze, PatternConfig, FRAMING, DISCLAIMER

VERSION = "problem-pattern-graph-v1"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/records.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/patterns-v1"


def resource_hashes():
    base = Path(__file__).parent
    return {str(p.relative_to(base)): digest(p) for p in [
        base / "normalization.py", base / "pattern_fingerprints.py", RESOURCE_PATH,
        *sorted((base / "resources").glob("*.json"))]}


def read_records(source=SOURCE):
    return sorted([CanonicalRecord.model_validate_json(line) for line in Path(source).read_text().splitlines() if line.strip()], key=lambda r: r.record_id)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def load_patterns(output=DEFAULT_OUTPUT, source=SOURCE):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["version"] != VERSION or manifest["source_sha256"] != digest(source) or manifest["resources"] != resource_hashes():
        raise ValueError("Pattern artifacts are stale; rebuild into a new versioned directory")
    for name, checksum in manifest["checksums"].items():
        if digest(output / name) != checksum:
            raise ValueError("Pattern artifact checksum mismatch: " + name)
    records = {r.record_id: r for r in read_records(source)}
    patterns = json.loads((output / "patterns.json").read_text())
    for pattern in patterns:
        if not set(pattern["record_ids"]) <= records.keys() or pattern["exclusion_reasons"] or pattern["size"] < 3:
            raise ValueError("Invalid pattern membership")
    return patterns, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "evaluation/patterns-v1")
    parser.add_argument("--cosine", type=float, default=.84)
    parser.add_argument("--distance", type=float, default=.16)
    parser.add_argument("--stability", type=float, default=.80)
    args = parser.parse_args()
    if args.output.exists() or args.report_dir.exists():
        parser.error("Output/report directories must be new: existing artifacts are never overwritten")
    start = time.perf_counter()
    records = read_records(args.source)
    name = "sentence-transformers/all-MiniLM-L6-v2"
    cached = try_to_load_from_cache(name, "config.json")
    if not isinstance(cached, str):
        parser.error("MiniLM is not cached locally. Prepare the existing retrieval model first.")
    snapshot = Path(cached).parent
    unique = sorted({r.problem_normalized for r in records})
    encoder = ProcessSentenceEncoder(str(snapshot))
    chunks = []
    try:
        for offset in range(0, len(unique), 128):
            chunks.append(encoder.encode(unique[offset:offset+128], normalize_embeddings=True))
            print(f"Encoded {min(offset+128, len(unique))}/{len(unique)} unique problems", flush=True)
    finally:
        encoder.close()
    encoded = dict(zip(unique, np.concatenate(chunks)))
    vectors = np.asarray([encoded[r.problem_normalized] for r in records], dtype=np.float32)
    embedding_seconds = time.perf_counter()-start
    config = PatternConfig(cosine=args.cosine, distance=args.distance, stability=args.stability)
    print("Building mutual-neighbour graph, clustering comparison and perturbations", flush=True)
    result = analyze(records, vectors, config)
    args.output.mkdir(parents=True, exist_ok=False)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    for key in ("nodes", "edges"):
        (args.output / f"{key}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in result[key]))
    write_json(args.output / "patterns.json", result["patterns"])
    write_json(args.output / "cluster_audit.json", result["audits"])
    write_json(args.output / "record_ids.json", [r.record_id for r in records])
    np.save(args.output / "problem_embeddings.npy", vectors)
    index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    faiss.write_index(index, str(args.output / "problem.faiss"))
    manifest = dict(version=VERSION, source_sha256=digest(args.source), record_count=len(records),
        fingerprint_version=FINGERPRINT_VERSION, resources=resource_hashes(), embedding_model=name,
        model_revision=snapshot.name, model_files={p.name:digest(p) for p in snapshot.iterdir() if p.suffix in {".safetensors", ".bin"}},
        embedding_input="problem_normalized only", config=asdict(config), prototype_count=result["prototype_count"],
        selection="complete_link", framing=FRAMING, disclaimer=DISCLAIMER,
        checksums={p.name:digest(p) for p in args.output.iterdir() if p.is_file()})
    write_json(args.output / "manifest.json", manifest)
    comparison = {}
    for method, rows in result["audits"].items():
        retained = [r for r in rows if not r["exclusion_reasons"]]
        comparison[method] = dict(candidate_clusters=len(rows), retained_clusters=len(retained),
            excluded_unstable=sum("unstable membership" in r["exclusion_reasons"] for r in rows),
            distributions={field:[r[field] for r in retained] for field in ("size", "mean_similarity", "minimum_similarity", "stability")})
    report = dict(framing=FRAMING, disclaimer=DISCLAIMER, human_review_status="Not reviewed; no automated correctness claim",
        record_count=len(records), edge_count=len(result["edges"]), prototype_count=result["prototype_count"],
        config=asdict(config), comparison=comparison, embedding_seconds=embedding_seconds,
        build_seconds=time.perf_counter()-start, artifact_bytes=sum(p.stat().st_size for p in args.output.iterdir()),
        selection_rationale="Complete-link bounds pairwise distance; connected components can chain unrelated endpoints. Selection is not a human quality validation.")
    write_json(args.report_dir / "report.json", report)
    samples = sorted(result["patterns"], key=lambda p: (-p["size"], p["pattern_id"]))[:10]
    samples += sorted(result["patterns"], key=lambda p: (p["stability"], p["pattern_id"]))[:10]
    samples = list({p["pattern_id"]:p for p in samples}.values())
    write_json(args.report_dir / "human_review.json", dict(framing=FRAMING, disclaimer=DISCLAIMER,
        sampling="10 largest plus 10 lowest-stability retained patterns, deduplicated; not a random quality estimate",
        instructions="Inspect original source records. Leave labels null until reviewed. Purity: 0–1; other labels: appropriate/inappropriate/uncertain; conflict: present/absent/uncertain.",
        samples=[dict(pattern=p, source_records=[records[i].model_dump(mode="json") for i in range(len(records)) if records[i].record_id in p["record_ids"]],
            review=dict(cluster_purity=None, label_appropriateness=None, side_location_conflict=None,
                        action_variant_appropriateness=None, reviewer=None, notes=None)) for p in samples]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
