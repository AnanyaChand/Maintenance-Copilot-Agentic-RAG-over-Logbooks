"""CSV schema detection and canonical MaintNet record ingestion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
from pathlib import Path
import re
import unicodedata

from .models import CanonicalRecord, IngestionReport
from .normalization import NormalizationResources, normalize_text


class IngestionError(ValueError):
    """Raised when a dataset cannot be safely mapped to the canonical schema."""


ALIASES: dict[str, set[str]] = {
    "record_id": {"id", "ident", "identifier", "recordid", "recordnumber", "workorder", "wo"},
    "problem": {"problem", "issue", "issueproblem", "fault", "discrepancy", "description"},
    "action": {"action", "correctiveaction", "resolution", "workperformed"},
    "ata_chapter": {"ata", "atacode", "atachapter", "chapter"},
    "opened_at": {"date", "opendate", "dateopened", "opened", "openedat"},
    "closed_at": {"closedate", "dateclosed", "closed", "closedat"},
}
REQUIRED_FIELDS = frozenset({"record_id", "problem", "action"})
DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d")


@dataclass(frozen=True)
class IngestionResult:
    records: list[CanonicalRecord]
    report: IngestionReport


def normalize_header(header: str) -> str:
    header = unicodedata.normalize("NFKC", header).lstrip("\ufeff").lower()
    return re.sub(r"[^a-z0-9]+", "", header)


def detect_schema(headers: list[str] | None) -> dict[str, str]:
    if not headers:
        raise IngestionError("CSV has no header row")

    mapping: dict[str, str] = {}
    for actual in headers:
        candidate = normalize_header(actual)
        matches = [canonical for canonical, aliases in ALIASES.items() if candidate in aliases]
        if len(matches) > 1:
            raise IngestionError(f"Ambiguous column {actual!r}: matches {matches}")
        if matches:
            canonical = matches[0]
            if canonical in mapping:
                raise IngestionError(
                    f"Multiple columns map to {canonical!r}: {mapping[canonical]!r} and {actual!r}"
                )
            mapping[canonical] = actual

    missing = sorted(REQUIRED_FIELDS - mapping.keys())
    if missing:
        raise IngestionError(
            f"Missing required canonical fields {missing}; received headers {headers}"
        )
    return mapping


def _parse_optional_date(value: str, field: str, row_number: int) -> date | None:
    value = value.strip()
    if not value:
        return None
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise IngestionError(f"Invalid {field} date {value!r} on CSV row {row_number}") from exc


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_csv(
    path: str | Path,
    *,
    resources: NormalizationResources | None = None,
) -> IngestionResult:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise IngestionError(f"Dataset does not exist or is not a file: {source_path}")

    resources = resources or NormalizationResources.load()
    records: list[CanonicalRecord] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    skipped_blank_rows = 0
    source_rows = 0

    try:
        handle = source_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise IngestionError(f"Could not open dataset {source_path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        mapping = detect_schema(reader.fieldnames)
        canonical_headers = set(mapping.values())

        for row_number, row in enumerate(reader, start=2):
            source_rows += 1
            if None in row:
                raise IngestionError(f"Malformed CSV row {row_number}: too many fields")
            if all(not (value or "").strip() for value in row.values()):
                skipped_blank_rows += 1
                continue

            def required_value(field: str) -> str:
                value = row.get(mapping[field]) or ""
                if not value.strip():
                    raise IngestionError(
                        f"Blank required field {field!r} on CSV row {row_number}"
                    )
                return value

            record_id = required_value("record_id").strip()
            if record_id in seen_ids:
                duplicate_ids.add(record_id)
                continue
            seen_ids.add(record_id)

            problem = required_value("problem")
            action = required_value("action")
            problem_normalized = normalize_text(problem, resources)
            action_normalized = normalize_text(action, resources)
            if not problem_normalized or not action_normalized:
                raise IngestionError(
                    f"Normalization produced blank text on CSV row {row_number}"
                )

            ata = (row.get(mapping.get("ata_chapter", "")) or "").strip() or None
            opened_at = _parse_optional_date(
                row.get(mapping.get("opened_at", "")) or "", "opened_at", row_number
            )
            closed_at = _parse_optional_date(
                row.get(mapping.get("closed_at", "")) or "", "closed_at", row_number
            )
            metadata = {
                header: (value or "")
                for header, value in row.items()
                if header not in canonical_headers
            }

            records.append(
                CanonicalRecord(
                    record_id=record_id,
                    problem_original=problem,
                    action_original=action,
                    problem_normalized=problem_normalized,
                    action_normalized=action_normalized,
                    ata_chapter=ata,
                    opened_at=opened_at,
                    closed_at=closed_at,
                    source_row=row_number,
                    metadata=metadata,
                )
            )

    if duplicate_ids:
        sample = ", ".join(sorted(duplicate_ids)[:10])
        raise IngestionError(f"Duplicate record IDs detected: {sample}")

    report = IngestionReport(
        source_path=str(source_path),
        source_sha256=_source_hash(source_path),
        encoding="utf-8-sig",
        detected_columns=mapping,
        source_rows=source_rows,
        loaded_records=len(records),
        skipped_blank_rows=skipped_blank_rows,
        duplicate_ids=[],
    )
    return IngestionResult(records=records, report=report)

