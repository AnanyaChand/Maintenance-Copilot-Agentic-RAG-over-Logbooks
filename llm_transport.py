"""OpenAI-compatible Chat Completions transport with a killable wall-clock limit."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Protocol

from .llm_config import LLMConfig


class ProviderFailure(Exception):
    def __init__(self, code="provider_error"):
        self.code = code
        super().__init__(code)


class ToolCallingProvider(Protocol):
    def complete(self, payload: dict, *, timeout: float) -> dict: ...


class OpenAICompatibleProvider:
    """No retries, redirect following, raw-error propagation or payload logging.

    A child process bounds even DNS/header/body stalls. Credentials travel only
    through stdin, never command-line arguments. The child is killed on timeout.
    """
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, payload: dict, *, timeout: float) -> dict:
        if timeout <= 0:
            raise ProviderFailure("timeout")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "maintenance_copilot.llm_http_worker"],
                input=json.dumps({"url": self.config.base_url + "/chat/completions",
                                  "key": self.config.api_key, "payload": payload, "timeout": timeout}),
                text=True, capture_output=True, timeout=timeout, check=False,
            )
            if result.returncode:
                raise ProviderFailure()
            envelope = json.loads(result.stdout)
            if envelope.get("error"):
                raise ProviderFailure(envelope["error"] if envelope["error"] in
                                      {"timeout", "malformed_output"} else "provider_error")
            if not isinstance(envelope.get("message"), dict):
                raise ProviderFailure("malformed_output")
            return envelope["message"]
        except subprocess.TimeoutExpired:
            raise ProviderFailure("timeout") from None
        except ProviderFailure:
            raise
        except Exception:
            raise ProviderFailure() from None
