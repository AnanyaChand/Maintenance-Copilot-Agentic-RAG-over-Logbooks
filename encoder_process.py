"""Persistent process isolation for sentence-transformer query encoding.

FAISS and PyTorch wheels can bundle incompatible OpenMP runtimes on macOS.
Keeping MiniLM in a child process prevents that native-library collision while
retaining exact FAISS search in the parent process.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from typing import Any, Sequence

import numpy as np


class EncoderProcessError(RuntimeError):
    pass


class ProcessSentenceEncoder:
    def __init__(self, model_name: str) -> None:
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "maintenance_copilot.embedding_worker",
                model_name,
                "--local-files-only",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=environment,
        )
        ready = self._read_message()
        if ready.get("status") != "ready":
            self.close()
            raise EncoderProcessError(ready.get("error", "embedding worker did not start"))
        atexit.register(self.close)

    def _read_message(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise EncoderProcessError("embedding worker stdout is unavailable")
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            raise EncoderProcessError(f"embedding worker exited unexpectedly with code {code}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise EncoderProcessError(f"invalid embedding worker response: {line!r}") from exc

    def encode(self, sentences: str | Sequence[str], **kwargs: Any) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        if self._process.stdin is None or self._process.poll() is not None:
            raise EncoderProcessError("embedding worker is not running")
        request = {
            "sentences": list(sentences),
            "normalize_embeddings": bool(kwargs.get("normalize_embeddings", True)),
        }
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        response = self._read_message()
        if response.get("status") != "ok":
            raise EncoderProcessError(response.get("error", "embedding worker failed"))
        return np.asarray(response["vectors"], dtype=np.float32)

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write('{"command":"close"}\n')
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=3)

