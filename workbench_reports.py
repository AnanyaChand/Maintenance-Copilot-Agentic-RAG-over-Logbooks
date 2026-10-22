"""Human-only metrics and reviewer agreement; no fallback to provisional labels."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from .workbench import Catalog, ReviewStore, DEFAULT_REVIEW_DIR, focused_pack


def fraction(n, d):
    return {"value": n / d if d else None, "numerator": n, "denominator": d}


def human_metrics(catalog, reviews):
    reviews = [r for r in reviews if has_judgments(r)]
    relevance, precision, grounds, refused = [], [], [], []
    claims, supported_claims = 0, 0
    labeled_sufficiency = []
    tiers = {}
    for review in reviews:
        case = catalog.cases[review.case_id]
        cited = set(case["answer"]["cited_record_ids"])
        for rid, judgment in review.records.items():
            if rid not in cited:
                continue
            if judgment.relevance in {"relevant", "partially_relevant", "irrelevant"}:
                relevance.append({"relevant": 1., "partially_relevant": .5, "irrelevant": 0.}[judgment.relevance])
            if judgment.historical_action_support in {"supports_answer", "partially_supports_answer", "does_not_support_answer"}:
                precision.append(int(judgment.historical_action_support == "supports_answer"))
        if review.answer_grounding in {"supported", "partially_supported", "unsupported"}:
            grounds.append(int(review.answer_grounding == "supported"))
        if review.meaningful_claim_count is not None and review.claims_with_supported_citations is not None:
            claims += review.meaningful_claim_count
            supported_claims += review.claims_with_supported_citations
        is_refusal = case["status"] == "insufficient_evidence"
        if is_refusal and review.refusal_correctness in {"correct_refusal", "incorrect_refusal"}:
            refused.append(int(review.refusal_correctness == "correct_refusal"))
        sufficient = review.evidence_sufficiency
        if sufficient in {None, "unsure"} and is_refusal:
            sufficient = {"correct_refusal": "insufficient", "incorrect_refusal": "sufficient"}.get(review.refusal_correctness)
        if sufficient in {"sufficient", "insufficient"}:
            labeled_sufficiency.append((is_refusal, sufficient == "insufficient"))
        tier = tiers.setdefault(case["tier"], Counter())
        tier["reviewed_cases"] += 1
        if review.safety_assessment:
            tier[review.safety_assessment] += 1
    should_refuse = [actual for actual, expected in labeled_sufficiency if expected]
    return {
        "reviewed_cases": len(reviews),
        "citation_relevance": fraction(sum(relevance), len(relevance)),
        "citation_precision": fraction(sum(precision), len(precision)),
        "citation_coverage": fraction(supported_claims, claims),
        "answer_grounding_precision": fraction(sum(grounds), len(grounds)),
        "refusal_precision": fraction(sum(refused), len(refused)),
        "refusal_recall": fraction(sum(should_refuse), len(should_refuse)),
        "false_confidence_rate": fraction(sum(not actual for actual in should_refuse), len(should_refuse)),
        "evidence_sufficiency_cases": len(labeled_sufficiency),
        "tier_safety": {tier: dict(counts) for tier, counts in tiers.items()},
        "definitions": {
            "citation_relevance": "Cited-record partial credit: relevant=1, partially_relevant=.5, irrelevant=0.",
            "citation_precision": "Strict human action-support precision over judged cited records; partial support counts zero. Not automated verbatim fidelity.",
            "citation_coverage": "Human-audited meaningful claims with supported citations / total audited claims. Missing counts remain unmeasured.",
            "answer_grounding_precision": "Fully supported / all decidably judged answers; partial support counts zero.",
            "refusal_recall": "Refusals / human-insufficient cases. Requires explicit sufficiency for answered cases; never inferred from provisional expectations.",
            "false_confidence_rate": "Answers / human-insufficient cases, irrespective of numerical confidence.",
        },
        "limitations": ["Judged subsets only; incomplete, unsure and not-applicable labels are excluded from relevant denominators.",
                        "Unreviewed cases are not successes. Individual reviewer judgments are not domain-expert certification.",
                        "If sufficiency is judged only on refused cases, recall is selection-biased; inspect evidence_sufficiency_cases and review answered cases too."]}


def agreement_stats(pairs):
    n = len(pairs)
    raw = sum(a == b for a, b in pairs) / n if n else None
    kappa = None
    if n >= 2:
        left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
        chance = sum(left[k] * right[k] for k in set(left) | set(right)) / n**2
        if chance < 1:
            kappa = (raw - chance) / (1 - chance)
    return {"n": n, "raw_agreement": raw, "cohens_kappa": kappa,
            "warning": "Small/non-independent sample; not a reliability certification." if n < 30 else "Case variants may be dependent; no reliability guarantee.",
            "undefined_reason": "No shared judgments, fewer than two pairs, or chance agreement equals one." if kappa is None else None}


def agreement_report(reviews):
    by_reviewer = {}
    for review in reviews:
        by_reviewer.setdefault(review.reviewer_name, {})[review.case_id] = review
    pairwise, disagreements = [], []
    for one, two in combinations(sorted(by_reviewer), 2):
        relevance_pairs, grounded_pairs, refusal_pairs = [], [], []
        raw_fields = {}
        for cid in sorted(by_reviewer[one].keys() & by_reviewer[two].keys()):
            a, b = by_reviewer[one][cid], by_reviewer[two][cid]
            checks = [(None, key, getattr(a, key), getattr(b, key)) for key in
                      ("answer_grounding", "refusal_correctness", "safety_assessment", "evidence_sufficiency")]
            for rid in sorted(a.records.keys() & b.records.keys()):
                checks.extend((rid, key, getattr(a.records[rid], key), getattr(b.records[rid], key)) for key in
                              ("relevance", "fault_match", "historical_action_support"))
            for rid, key, av, bv in checks:
                if av is None or bv is None:
                    continue
                if av != bv:
                    disagreements.append({"case_id": cid, "record_id": rid, "field": key,
                        "reviewer_a": one, "reviewer_b": two, "label_a": av, "label_b": bv})
                if av in {"unsure", "not_applicable", "no_answer_to_review"} or bv in {"unsure", "not_applicable", "no_answer_to_review"}:
                    continue
                raw_fields.setdefault(key, []).append((av, bv))
                if key == "relevance":
                    relevance_pairs.append((int(av != "irrelevant"), int(bv != "irrelevant")))
                if key == "answer_grounding":
                    grounded_pairs.append((int(av != "unsupported"), int(bv != "unsupported")))
                if key == "refusal_correctness":
                    refusal_pairs.append((av, bv))
        pairwise.append({"reviewers": [one, two], "collapsed_relevance": agreement_stats(relevance_pairs),
                         "grounded_vs_unsupported": agreement_stats(grounded_pairs),
                         "refusal_correctness": agreement_stats(refusal_pairs),
                         "raw_by_field": {k: agreement_stats(pairs) for k, pairs in raw_fields.items()}})
    return {"pairwise": pairwise, "disagreement_queue": disagreements,
            "collapse_policy": "Relevance: relevant+partially_relevant versus irrelevant. Grounding: supported+partially_supported versus unsupported. Unsure/NA/unreviewed excluded. No majority-vote relabeling."}


def has_judgments(review):
    return (any(getattr(review, k) is not None for k in ("answer_grounding", "refusal_correctness", "safety_assessment", "evidence_sufficiency"))
            or any(any(value is not None for value in labels.model_dump().values()) for labels in review.records.values())
            or (review.meaningful_claim_count is not None and review.claims_with_supported_citations is not None))


def report(catalog, store):
    reviews = [r for r, _ in store.all()]
    names = sorted({r.reviewer_name for r in reviews if has_judgments(r)})
    return {"dataset_fingerprint": catalog.fingerprint,
            "provisional_automated_labels": {"provenance": "assistant-authored, not human validated",
                "development": catalog.report["development"]["metrics"], "held_out": catalog.report["held_out"]["metrics"]},
            "human_reviewed_labels": {name: {
                "all_reviewed": human_metrics(catalog, [r for r in reviews if r.reviewer_name == name]),
                "by_split": {split: human_metrics(catalog, [r for r in reviews if r.reviewer_name == name and catalog.cases[r.case_id]["split"] == split])
                             for split in ("development", "held_out")}} for name in names} or None,
            "agreement": agreement_report(reviews),
            "progress": {name: {"touched": sum(r.reviewer_name == name for r in reviews),
                "complete": sum(r.reviewer_name == name and catalog.complete(r) for r in reviews), "total": len(catalog.cases)} for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "pack", "export", "import"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--actor")
    args = parser.parse_args()
    catalog = Catalog()
    store = ReviewStore(args.directory, catalog)
    if args.command == "import":
        if not args.input or not args.actor:
            parser.error("import needs --input and --actor")
        print(json.dumps({"saved_revisions": store.import_csv(args.input.read_text(), args.actor)}))
        return
    value = store.export_csv() if args.command == "export" else json.dumps(
        focused_pack(catalog) if args.command == "pack" else report(catalog, store), indent=2)
    if args.output:
        if args.output.resolve() in catalog.protected:
            raise ValueError("Cannot overwrite a protected source file")
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(value)
    else:
        print(value)


if __name__ == "__main__":
    main()
