"""JSON-lines worker hosting MiniLM without importing FAISS."""

from __future__ import annotations

import argparse
import json
import sys


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            args.model_name,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        _send({"status": "error", "error": f"Could not load embedding model: {exc}"})
        return 1

    _send({"status": "ready"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                return 0
            vectors = model.encode(
                request["sentences"],
                normalize_embeddings=bool(request.get("normalize_embeddings", True)),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            _send({"status": "ok", "vectors": vectors.tolist()})
        except Exception as exc:
            _send({"status": "error", "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

