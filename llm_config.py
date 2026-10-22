"""Explicit opt-in configuration. Secrets never enter public mode metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from dotenv import dotenv_values


@dataclass(frozen=True)
class LLMConfig:
    requested: bool = False
    enabled: bool = False
    base_url: str = field(default="", repr=False)
    api_key: str = field(default="", repr=False)
    model: str = ""
    timeout: float = 30.0
    problem: str | None = None

    @classmethod
    def from_env(cls, environ=None):
        env = ({**dotenv_values(Path(__file__).resolve().parents[2] / ".env"), **os.environ}
               if environ is None else environ)
        mode = str(env.get("COPILOT_MODE", "deterministic")).strip().lower()
        if mode == "deterministic":
            return cls()
        if mode != "llm":
            return cls(requested=True, problem="invalid_configuration")
        if env.get("RETRIEVAL_VARIANT", "baseline") != "baseline":
            return cls(requested=True, problem="invalid_configuration")
        if str(env.get("LLM_ENABLED", "false")).strip().lower() != "true":
            return cls(requested=True, problem="llm_disabled")
        url, key, model = (str(env.get(k) or "").strip() for k in
                           ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
        if not all((url, key, model)):
            return cls(requested=True, problem="missing_configuration")
        try:
            parsed = urlsplit(url)
            valid_url = (parsed.scheme == "https" or
                         (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}))
            valid_url = valid_url and bool(parsed.hostname) and not any(
                (parsed.username, parsed.password, parsed.query, parsed.fragment))
            valid_model = bool(re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", model)) and "://" not in model and key not in model
            timeout = float(env.get("LLM_TIMEOUT_SECONDS", "30"))
            if not (valid_url and valid_model and math.isfinite(timeout) and 0 < timeout <= 120):
                raise ValueError
            if any(ord(c) < 32 or ord(c) == 127 for c in url + key):
                raise ValueError
        except (ValueError, TypeError):
            return cls(requested=True, problem="invalid_configuration")
        return cls(requested=True, enabled=True, base_url=url.rstrip("/"), api_key=key,
                   model=model, timeout=timeout)


REASONS = {
    "missing_configuration": "Required LLM configuration is missing.",
    "invalid_configuration": "LLM configuration is invalid.",
    "llm_disabled": "LLM_ENABLED is not true.",
    "provider_error": "The provider request failed.",
    "timeout": "The LLM time budget expired.",
    "malformed_output": "The provider returned an invalid structured response.",
    "invalid_tool_sequence": "The provider did not follow the allowed tool sequence.",
    "grounding_failed": "Generated claims failed source or citation verification.",
    "metadata_mismatch": "Generated evidence or metadata did not match the executed tools.",
    "tool_error": "An LLM-requested evidence tool failed.",
    "retrieval_changed": "Retrieval support changed during the tool workflow.",
}
