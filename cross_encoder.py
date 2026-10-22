"""Local-only, isolated cross-encoder; logits are ranking scores, not probabilities."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts/cross-encoder"


class LocalCrossEncoder:
    def __init__(self, path=MODEL_PATH):
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [sys.executable, "-m", __name__, str(path)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1,
            env={**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
        atexit.register(self.close)
        try:
            self._read(60)
        except Exception:
            self.close()
            raise

    def _read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("Cross-encoder timed out; no evidence approved")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Cross-encoder exited; no evidence approved")
        message = json.loads(line)
        if message.get("status") == "error":
            raise RuntimeError(message["error"])
        return message

    def predict(self, pairs):
        with self.lock:
            try:
                self.process.stdin.write(json.dumps({"pairs": pairs}) + "\n")
                self.process.stdin.flush()
                return self._read(30)["logits"]
            except Exception:
                self.close()
                raise

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"close"}\n')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self.process.wait(timeout=3)


def worker():
    def send(value):
        print(json.dumps(value), flush=True)
    try:
        import torch
        from sentence_transformers import CrossEncoder
        torch.set_num_threads(1)
        model = CrossEncoder(sys.argv[1], device="cpu", max_length=256, local_files_only=True)
        send({"status": "ready"})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("command") == "close":
                return
            scores = model.predict(request["pairs"], activation_fn=torch.nn.Identity(),
                                   batch_size=16, show_progress_bar=False)
            send({"status": "ok", "logits": scores.tolist()})
    except Exception as exc:
        send({"status": "error", "error": str(exc)})


if __name__ == "__main__":
    worker()
