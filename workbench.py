"""Offline review catalog and transactional, revisioned human annotations.

No retrieval calls, model downloads, or production configuration writes occur here.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .indexing import load_canonical_records
from .review import digest as original_digest
from .tools import PROJECT_ROOT

EVALUATION = PROJECT_ROOT / "evaluation"
ORIGINAL_CSV = EVALUATION / "phase6-v1-final/review.csv"
DEFAULT_REVIEW_DIR = EVALUATION / "reviews"
RECORD_FIELDS = {
    "relevance": ["relevant", "partially_relevant", "irrelevant", "unsure"],
    "fault_match": ["exact_or_near_exact", "same_component_different_symptom", "related_but_weak", "mismatch", "unsure"],
    "historical_action_support": ["supports_answer", "partially_supports_answer", "does_not_support_answer", "not_applicable", "unsure"],
}
CASE_FIELDS = {
    "answer_grounding": ["supported", "partially_supported", "unsupported", "no_answer_to_review"],
    "refusal_correctness": ["correct_refusal", "incorrect_refusal", "not_applicable"],
    "safety_assessment": ["safe", "needs_limitation", "unsafe_overclaim", "unsure"],
    "evidence_sufficiency": ["sufficient", "insufficient", "unsure"],
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevance: Literal["relevant", "partially_relevant", "irrelevant", "unsure"] | None = None
    fault_match: Literal["exact_or_near_exact", "same_component_different_symptom", "related_but_weak", "mismatch", "unsure"] | None = None
    historical_action_support: Literal["supports_answer", "partially_supports_answer", "does_not_support_answer", "not_applicable", "unsure"] | None = None


class CaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    reviewer_name: str = Field(min_length=1, max_length=120)
    review_origin: Literal["human"] = "human"
    answer_grounding: Literal["supported", "partially_supported", "unsupported", "no_answer_to_review"] | None = None
    refusal_correctness: Literal["correct_refusal", "incorrect_refusal", "not_applicable"] | None = None
    safety_assessment: Literal["safe", "needs_limitation", "unsafe_overclaim", "unsure"] | None = None
    evidence_sufficiency: Literal["sufficient", "insufficient", "unsure"] | None = None
    meaningful_claim_count: int | None = Field(default=None, ge=0, le=10000)
    claims_with_supported_citations: int | None = Field(default=None, ge=0, le=10000)
    reviewer_notes: str = Field(default="", max_length=20000)
    suggested_expected_record_ids: list[str] = Field(default_factory=list)
    reviewed_at: datetime | None = None
    records: dict[str, RecordReview] = Field(default_factory=dict)

    @field_validator("reviewer_name")
    @classmethod
    def clean_name(cls, value):
        value = value.strip()
        if not value or any(ord(c) < 32 for c in value):
            raise ValueError("Enter a reviewer name or stable pseudonym")
        return value

    @model_validator(mode="after")
    def count_consistency(self):
        if (self.meaningful_claim_count is not None and self.claims_with_supported_citations is not None
                and self.claims_with_supported_citations > self.meaningful_claim_count):
            raise ValueError("Supported-citation count exceeds total meaningful claims")
        if len(set(self.suggested_expected_record_ids)) != len(self.suggested_expected_record_ids):
            raise ValueError("Duplicate suggested record IDs")
        return self


class Catalog:
    def __init__(self, root=PROJECT_ROOT):
        root = Path(root)
        ev = root / "evaluation"
        self.original_csv = ev / "phase6-v1-final/review.csv"
        sources = [self.original_csv, ev / "phase6-v1-final/report.json", ev / "benchmark.v1.json",
                   ev / "results/advanced.baseline.v1.json", ev / "results/advanced.field-gated.v1.json",
                   root / "data/processed/records.jsonl"]
        self.source_hashes = {str(p.relative_to(root)): file_hash(p) for p in sources}
        self.fingerprint = hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()
        self.protected = {p.resolve() for p in sources}
        self.records = {r.record_id: r for r in load_canonical_records(sources[-1])}
        self.report = json.loads(sources[1].read_text())
        suite = json.loads(sources[2].read_text())
        self.labels = {c["id"]: c for c in suite["cases"]}
        self.baseline = {c["id"]: c for c in json.loads(sources[3].read_text())["cases"]}
        self.gated = {c["id"]: c for c in json.loads(sources[4].read_text())["cases"]}
        pools = {}
        with self.original_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if original_digest(row) != row["item_sha256"]:
                    raise ValueError("Provisional CSV source checksum mismatch")
                cid, rid = row["case_id"], row["record_id"]
                if cid not in self.labels or (rid and rid not in self.records):
                    raise ValueError("Unknown case/record in provisional CSV")
                pools.setdefault(cid, set()).update([rid] if rid else [])
        self.cases = {}
        for case in self.report["cases"]:
            cid = case["id"]
            case = dict(case)
            case["split"] = "development" if cid in self.report["split"]["development"] else "held_out"
            # Review exactly the exported candidate pool plus any actual citations;
            # do not turn all top-50 nearest neighbors into obligatory annotations.
            case["pool"] = sorted(pools.get(cid, set()) | set(case["answer"]["cited_record_ids"]))
            case["category"] = self.labels[cid]["category"]
            case["expected_ids"] = self.labels[cid]["expected_relevant_record_ids"]
            case["suspected_false_confidence"] = bool(case["expected_refusal"] and case["status"] == "answered")
            self.cases[cid] = case

    def validate(self, review):
        if review.case_id not in self.cases:
            raise ValueError("Unknown query ID")
        case = self.cases[review.case_id]
        if not set(review.records).issubset(case["pool"]):
            raise ValueError("Invalid record ID for this case's review pool")
        if not set(review.suggested_expected_record_ids).issubset(self.records):
            raise ValueError("Invalid suggested expected record ID")
        refused = case["status"] == "insufficient_evidence"
        if not refused and review.refusal_correctness not in {None, "not_applicable"}:
            raise ValueError("Correct/incorrect refusal applies only to an actual refusal")
        if not refused and review.answer_grounding == "no_answer_to_review":
            raise ValueError("An answer exists for this case")
        if refused and review.answer_grounding not in {None, "no_answer_to_review"}:
            raise ValueError("Refused case has no maintenance answer to judge")
        if refused and review.refusal_correctness == "not_applicable":
            raise ValueError("Refusal correctness applies to this refused case")
        if ((review.refusal_correctness == "correct_refusal" and review.evidence_sufficiency == "sufficient")
                or (review.refusal_correctness == "incorrect_refusal" and review.evidence_sufficiency == "insufficient")):
            raise ValueError("Refusal and evidence sufficiency judgments conflict")

    def complete(self, review):
        if review is None:
            return False
        return (all(getattr(review, k) is not None for k in list(CASE_FIELDS)[:3])
                and all(rid in review.records and all(getattr(review.records[rid], k) is not None
                    for k in RECORD_FIELDS) for rid in self.cases[review.case_id]["pool"]))

    def evidence(self, cid, rid):
        case = self.cases[cid]
        features = self.report["development_features"].get(cid)
        feature = next((c for c in (features or {}).get("candidates", []) if c["record_id"] == rid), None)
        hit, origin = None, "not retained for this record"
        for label, candidates in [
            ("selective feature extraction", (features or {}).get("response", {}).get("results", [])),
            ("selective answer", case["answer"]["evidence"]),
            ("archived old-gated answer", self.gated[cid]["answer"]["evidence"]),
            ("archived baseline answer", self.baseline[cid]["answer"]["evidence"]),
        ]:
            hit = next((h for h in candidates if h["record_id"] == rid), None)
            if hit:
                origin = label
                break
        retrieved_ids = set(self.baseline[cid]["rankings"]["hybrid"]) | set(self.gated[cid]["rankings"]["hybrid"])
        if rid in case["answer"]["cited_record_ids"]:
            role = "Cited evidence"
        elif case["status"] == "insufficient_evidence" and rid in retrieved_ids:
            role = "Nearest candidate — excluded from this refused answer"
        elif rid in retrieved_ids or feature:
            role = "Retrieved but uncited evidence"
        else:
            role = "Provisional review-pool record — retrieval provenance unavailable"
        return {"record": self.records[rid], "hit": hit or {}, "feature": feature or {},
                "score_origin": origin, "role": role}


def filter_cases(catalog, reviews, *, status="all", tier="all", split="all", search="", flags=(), pack=None):
    result = []
    for cid, case in sorted(catalog.cases.items()):
        if pack is not None and cid not in pack:
            continue
        if status != "all" and case["status"] != status:
            continue
        if tier != "all" and case["tier"] != tier:
            continue
        if split != "all" and case["split"] != split:
            continue
        if "unreviewed" in flags and catalog.complete(reviews.get(cid)):
            continue
        if "high_confidence" in flags and case["answer"]["confidence"]["score"] < .8:
            continue
        if "suspected_false_confidence" in flags and not case["suspected_false_confidence"]:
            continue
        if "action_divergence" in flags and not case["decision"]["action_divergence"]:
            continue
        haystack = " ".join([cid, case["answer"]["query"], *case["pool"]]).casefold()
        if search.strip().casefold() not in haystack:
            continue
        result.append(cid)
    return result


def focused_pack(catalog):
    predicates = {
        "strong_supported": lambda c: c["tier"] == "TIER_A_STRONG_EVIDENCE" and not c["expected_refusal"],
        "tier_b_limited": lambda c: c["tier"] == "TIER_B_SUPPORTED_WITH_LIMITATIONS",
        "refusals": lambda c: c["status"] == "insufficient_evidence",
        "potential_false_confidence_high_score": lambda c: c["suspected_false_confidence"] or c["answer"]["confidence"]["score"] >= .8,
        "abbreviation_typo": lambda c: c["category"] in {"abbreviation", "noisy"},
        "action_divergence": lambda c: c["decision"]["action_divergence"],
    }
    candidates = {name: sorted([cid for cid, c in catalog.cases.items() if pred(c)],
                  key=lambda cid: (not catalog.cases[cid]["suspected_false_confidence"], cid)) for name, pred in predicates.items()}
    slots = [(name, i) for name in sorted(candidates, key=lambda k: len(candidates[k]))
             for i in range(min(10, len(candidates[name])))]
    assigned = {}
    def augment(slot, seen):
        for cid in candidates[slot[0]]:
            if cid in seen:
                continue
            seen.add(cid)
            if cid not in assigned or augment(assigned[cid], seen):
                assigned[cid] = slot
                return True
        return False
    for slot in slots:
        augment(slot, set())
    return {"dataset_fingerprint": catalog.fingerprint, "provisional_selection_only": True,
            "cases": {cid: slot[0] for cid, slot in sorted(assigned.items())},
            "composition": {name: {"requested": 10, "available": len(candidates[name]),
                "selected": sum(slot[0] == name for slot in assigned.values())} for name in predicates}}


class ReviewConflict(ValueError):
    pass


EXPORT_FIELDS = ["schema_version", "dataset_fingerprint", "case_id", "reviewer_name", "revision",
                 "row_kind", "record_id", "review_origin", *CASE_FIELDS,
                 "meaningful_claim_count", "claims_with_supported_citations", "reviewer_notes",
                 "suggested_expected_record_ids", "reviewed_at", *RECORD_FIELDS]


def csv_safe(value):
    value = "" if value is None else str(value)
    # Reversible escaping for spreadsheet formula injection and leading apostrophes.
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r", "'")) else value


def csv_unsafe(value):
    return value[1:] if value.startswith("'") and len(value) > 1 and value[1] in "=+-@\t\r'" else value


class ReviewStore:
    def __init__(self, directory, catalog):
        self.directory = Path(directory).resolve()
        self.catalog = catalog
        if self.directory in {p.parent for p in catalog.protected}:
            raise ValueError("Human reviews need a separate directory from source artifacts")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "human_review_v1.sqlite3"
        if self.db_path.is_symlink():
            raise ValueError("Review database cannot be a symbolic link")
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (reviewer TEXT, case_id TEXT, revision INTEGER NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(reviewer,case_id));
                CREATE TABLE IF NOT EXISTS audit (event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor TEXT NOT NULL, case_id TEXT NOT NULL, kind TEXT NOT NULL,
                    server_at TEXT NOT NULL, before_payload TEXT, after_payload TEXT);
            """)
            prior = db.execute("SELECT value FROM metadata WHERE key='dataset'").fetchone()
            if prior and prior[0] != catalog.fingerprint:
                raise ValueError("Review database belongs to a different evidence snapshot")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('dataset',?)", (catalog.fingerprint,))

    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=20)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    @contextmanager
    def locked(self):
        with (self.directory / ".review.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def all(self):
        with self.connect() as db:
            return [(CaseReview.model_validate_json(payload), revision) for payload, revision in
                    db.execute("SELECT payload,revision FROM reviews ORDER BY reviewer,case_id")]

    def load(self, reviewer, cid):
        with self.connect() as db:
            row = db.execute("SELECT payload,revision FROM reviews WHERE reviewer=? AND case_id=?", (reviewer, cid)).fetchone()
        return (CaseReview.model_validate_json(row[0]), row[1]) if row else (None, 0)

    def audit(self, reviewer=None):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            query = "SELECT * FROM audit" + (" WHERE actor=?" if reviewer else "") + " ORDER BY event_id"
            return [dict(r) for r in db.execute(query, (reviewer,) if reviewer else ())]

    def save(self, review, expected_revision):
        return self.save_many([(review, expected_revision)], actor=review.reviewer_name)[0]

    def save_many(self, items, actor, *, kind="edit"):
        items = [(CaseReview.model_validate(review.model_dump()), revision) for review, revision in items]
        if not actor.strip():
            raise ValueError("Audit actor is required")
        for review, _ in items:
            self.catalog.validate(review)
        if len({(r.reviewer_name, r.case_id) for r, _ in items}) != len(items):
            raise ValueError("Duplicate reviewer/case in transaction")
        now = datetime.now(timezone.utc).isoformat()
        with self.locked(), self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = []
            for review, revision in items:
                prior = db.execute("SELECT payload,revision FROM reviews WHERE reviewer=? AND case_id=?",
                                   (review.reviewer_name, review.case_id)).fetchone()
                if (prior[1] if prior else 0) != revision:
                    db.rollback()
                    # Conflict is auditable, while no annotation from this batch changes.
                    db.execute("INSERT INTO audit(actor,case_id,kind,server_at,before_payload,after_payload) VALUES (?,?,?,?,?,?)",
                        (actor, review.case_id, "revision_conflict", now, prior[0] if prior else None, review.model_dump_json()))
                    db.commit()
                    raise ReviewConflict("This review changed in another session; reload before editing/importing")
                current.append(prior)
            revisions = []
            for (review, revision), prior in zip(items, current):
                review = review.model_copy(update={"reviewed_at": review.reviewed_at or datetime.now(timezone.utc)})
                payload = review.model_dump_json()
                db.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?)",
                           (review.reviewer_name, review.case_id, revision + 1, payload))
                db.execute("INSERT INTO audit(actor,case_id,kind,server_at,before_payload,after_payload) VALUES (?,?,?,?,?,?)",
                           (actor, review.case_id, kind, now, prior[0] if prior else None, payload))
                revisions.append(revision + 1)
            db.commit()
            self._snapshot()
            return revisions

    def export_csv(self, reviewer=None):
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        for review, revision in self.all():
            if reviewer and review.reviewer_name != reviewer:
                continue
            base = {"schema_version": "human-review-v1", "dataset_fingerprint": self.catalog.fingerprint,
                    "case_id": review.case_id, "reviewer_name": review.reviewer_name, "revision": revision,
                    "review_origin": "human", **{k: getattr(review, k) for k in CASE_FIELDS},
                    "meaningful_claim_count": review.meaningful_claim_count,
                    "claims_with_supported_citations": review.claims_with_supported_citations,
                    "reviewer_notes": review.reviewer_notes,
                    "suggested_expected_record_ids": json.dumps(review.suggested_expected_record_ids),
                    "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else ""}
            writer.writerow({k: csv_safe(v) for k, v in dict(base, row_kind="case").items()})
            for rid, labels in sorted(review.records.items()):
                writer.writerow({k: csv_safe(v) for k, v in dict(base, row_kind="record", record_id=rid,
                                                                **labels.model_dump()).items()})
        return text.getvalue()

    def _snapshot(self):
        snapshots = self.directory / "snapshots"
        snapshots.mkdir(exist_ok=True)
        with self.connect() as db:
            event_id = db.execute("SELECT COALESCE(MAX(event_id),0) FROM audit").fetchone()[0]
        data = self.export_csv()
        path = snapshots / f"human_review_v1_{event_id:08d}.csv"
        if path.exists():
            if path.read_text() != data:
                raise ReviewConflict("Snapshot version already exists with different contents")
        else:
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        current = self.directory / "human_review_v1.csv"
        if current.is_symlink() or current.resolve() in self.catalog.protected:
            raise ValueError("Refusing to overwrite a protected source path")
        temporary = self.directory / ".human_review_v1.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, current)

    def import_csv(self, text, actor):
        if len(text.encode()) > 10_000_000:
            raise ValueError("Import exceeds 10 MB")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames != EXPORT_FIELDS:
            raise ValueError("Use a workbench export; provisional/Phase 6 CSVs are not human-review imports")
        grouped = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed CSV row")
            row = {k: csv_unsafe(v) for k, v in row.items()}
            if row["dataset_fingerprint"] != self.catalog.fingerprint or row["schema_version"] != "human-review-v1" or row["review_origin"] != "human":
                raise ValueError("Wrong dataset, schema or annotation provenance")
            key = (row["reviewer_name"], row["case_id"])
            base = {k: row[k] or None for k in CASE_FIELDS}
            base.update(case_id=row["case_id"], reviewer_name=row["reviewer_name"], reviewer_notes=row["reviewer_notes"],
                        suggested_expected_record_ids=json.loads(row["suggested_expected_record_ids"]),
                        reviewed_at=row["reviewed_at"] or None,
                        meaningful_claim_count=int(row["meaningful_claim_count"]) if row["meaningful_claim_count"] else None,
                        claims_with_supported_citations=int(row["claims_with_supported_citations"]) if row["claims_with_supported_citations"] else None)
            group = grouped.setdefault(key, {"base": base, "revision": int(row["revision"]), "case_seen": False, "records": {}})
            if group["base"] != base or group["revision"] != int(row["revision"]):
                raise ReviewConflict("Conflicting case-level fields within imported rows")
            if row["row_kind"] == "case" and not row["record_id"] and not any(row[k] for k in RECORD_FIELDS):
                if group["case_seen"]:
                    raise ReviewConflict("Duplicate case row")
                group["case_seen"] = True
            elif row["row_kind"] == "record" and row["record_id"]:
                if row["record_id"] in group["records"]:
                    raise ReviewConflict("Duplicate record row")
                group["records"][row["record_id"]] = {k: row[k] or None for k in RECORD_FIELDS}
            else:
                raise ValueError("Invalid row kind")
        items = []
        for group in grouped.values():
            if not group["case_seen"]:
                raise ValueError("Missing case row")
            review = CaseReview(**group["base"], records=group["records"])
            self.catalog.validate(review)
            current, revision = self.load(review.reviewer_name, review.case_id)
            if current and current == review:
                continue  # Idempotent reimport of unchanged exports.
            if current and revision != group["revision"]:
                raise ReviewConflict("Stale import would overwrite newer work")
            if current and not set(current.records).issubset(review.records):
                raise ReviewConflict("Import omitted existing record annotations; clear labels explicitly instead of removing rows")
            items.append((review, revision if current else 0))
        return self.save_many(items, actor=actor, kind="import") if items else []
