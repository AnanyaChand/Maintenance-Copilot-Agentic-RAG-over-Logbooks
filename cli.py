"""Command-line interface for validated MaintNet ingestion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from .ingestion import IngestionError, ingest_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "records.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "records.report.json"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintenance-copilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="validate and normalize a MaintNet CSV")
    ingest.add_argument("--dataset", type=Path, help="source CSV; defaults to DATASET_PATH")
    ingest.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build = subparsers.add_parser("build-index", help="build persistent BM25 and FAISS indexes")
    build.add_argument("--records", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    build.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    search = subparsers.add_parser("search", help="run hybrid retrieval")
    search.add_argument("query")
    search.add_argument("--domain", default="aviation")
    search.add_argument("--top-k", type=int, default=5)
    answer = subparsers.add_parser("answer", help="return cited historical evidence and tool trace")
    answer.add_argument("query")
    answer.add_argument("--top-k", type=int, default=5)
    return parser


def _run_ingest(dataset: Path | None, output: Path) -> int:
    configured = dataset or (Path(os.environ["DATASET_PATH"]) if os.getenv("DATASET_PATH") else None)
    if configured is None:
        print("DATASET_PATH is required (or pass --dataset)", file=sys.stderr)
        return 2

    try:
        result = ingest_csv(configured)
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in result.records:
            handle.write(record.model_dump_json() + "\n")

    report_path = output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(result.report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": result.report.loaded_records,
                "output": str(output),
                "report": str(report_path),
                "source_sha256": result.report.source_sha256,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        return _run_ingest(args.dataset, args.output)
    if args.command == "build-index":
        from .indexing import build_hybrid_index

        manifest = build_hybrid_index(args.records, args.report, args.artifacts)
        print(manifest.model_dump_json(indent=2))
        return 0
    if args.command == "search":
        from .tools import search_records

        response = search_records(args.query, domain=args.domain, top_k=args.top_k)
        print(response.model_dump_json(indent=2))
        return 0
    if args.command == "answer":
        from .agent import AgentExecutionError, GroundingError, answer_query
        try:
            response = answer_query(args.query, top_k=args.top_k)
        except (AgentExecutionError, GroundingError) as exc:
            print(json.dumps({"status": "error", "error": str(exc), "tool_trace": [
                event.model_dump(mode="json") for event in getattr(exc, "tool_trace", [])
            ]}), file=sys.stderr)
            return 1
        print(response.model_dump_json(indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

