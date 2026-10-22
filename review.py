"""CSV review round-trip. Empty forms never count as human validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .evaluation import write_new

FIELDS = ["case_id", "query", "tier", "record_id", "problem", "action", "answer",
          "is_cited",
          "relevance", "grounding", "refusal", "review_origin", "reviewer", "reviewed_at", "notes", "item_sha256"]
LABELS = {"relevance": {"", "relevant", "partially relevant", "irrelevant"},
          "grounding": {"", "supported", "partially supported", "unsupported"},
          "refusal": {"", "correct", "incorrect"}}


def digest(row):
    return hashlib.sha256(json.dumps([row[k] for k in FIELDS[:8]], ensure_ascii=False).encode()).hexdigest()


def safe_cell(value):
    text = str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


def export_review(report, records, path, candidates=None):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for case in report["cases"]:
            answer = case["answer"]
            # Unlabeled candidate pool supports review of refusals and unjudged
            # retrieval; being present in this CSV does not assert relevance.
            ids = sorted(set(answer["cited_record_ids"]) | set((candidates or {}).get(case["id"], []))) or [""]
            for rid in ids:
                record = records.get(rid)
                row = {key: "" for key in FIELDS}
                row.update(case_id=case["id"], query=answer["query"], tier=case["tier"], record_id=rid,
                           problem=record.problem_original if record else "",
                           action=record.action_original if record else "", answer=answer["answer_text"],
                           is_cited="yes" if rid in answer["cited_record_ids"] else "no")
                row = {key: safe_cell(value) for key, value in row.items()}
                row["item_sha256"] = digest(row)
                writer.writerow(row)


def reviewed_metrics(path, original):
    with original.open(newline="", encoding="utf-8") as handle:
        expected = {(r["case_id"], r["record_id"]): r for r in csv.DictReader(handle)}
    reviewed = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["case_id"], row["record_id"])
            if key in seen or key not in expected:
                raise ValueError("Unknown or duplicate annotation item")
            seen.add(key)
            if row["item_sha256"] != expected[key]["item_sha256"] or digest(row) != expected[key]["item_sha256"]:
                raise ValueError("Review source fields changed; labels must refer to the exported evidence")
            for field, allowed in LABELS.items():
                if row[field] not in allowed:
                    raise ValueError(f"Invalid {field} annotation")
            if row["review_origin"] == "human" and any(row[k] for k in LABELS):
                reviewed.append(row)
            elif row["review_origin"] not in {"", "human"}:
                raise ValueError("Only explicitly human annotations count as human review")
    if not reviewed:
        return None
    relevant = [r for r in reviewed if r["record_id"] and r["relevance"] and r["is_cited"] == "yes"]
    grouped = {}
    for row in reviewed:
        group = grouped.setdefault(row["case_id"], {})
        for key in ("grounding", "refusal"):
            if row[key]:
                if key in group and group[key] != row[key]:
                    raise ValueError("Conflicting query-level annotations across records")
                group[key] = row[key]
    grounds = [r["grounding"] for r in grouped.values() if r.get("grounding")]
    refusals = [r["refusal"] for r in grouped.values() if r.get("refusal")]
    return {
        "reviewed_cases": len(grouped), "relevance_judgments": len(relevant),
        "citation_relevance_partial_credit": sum({"relevant": 1., "partially relevant": .5, "irrelevant": 0.}[r["relevance"]]
                                                  for r in relevant) / len(relevant) if relevant else None,
        "answer_grounding_partial_credit": sum({"supported": 1., "partially supported": .5, "unsupported": 0.}[v]
                                               for v in grounds) / len(grounds) if grounds else None,
        "refusal_correctness": sum(v == "correct" for v in refusals) / len(refusals) if refusals else None,
        "grounding_judgments": len(grounds), "refusal_judgments": len(refusals),
        "annotations": reviewed,
        "limitation": "Human-reviewed subset only; partial credit and denominators differ from provisional qrel metrics."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    write_new(args.output, {"provisional_metrics": report["held_out"]["metrics"],
                           "human_validated_metrics": reviewed_metrics(args.annotations, args.original)})


if __name__ == "__main__":
    main()
