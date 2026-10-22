"""Conservative, dictionary-backed normalization for maintenance text."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata


DEFAULT_RESOURCES_DIR = Path(__file__).with_name("resources")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


class NormalizationResourceError(ValueError):
    """Raised when an editable normalization dictionary is invalid."""


@dataclass(frozen=True)
class NormalizationResources:
    abbreviations: dict[str, str]
    typo_corrections: dict[str, str]

    @classmethod
    def load(cls, directory: Path = DEFAULT_RESOURCES_DIR) -> "NormalizationResources":
        return cls(
            abbreviations=_load_dictionary(directory / "abbreviations.json"),
            typo_corrections=_load_dictionary(directory / "typo_corrections.json"),
        )


@dataclass(frozen=True)
class AppliedReplacement:
    original: str
    replacement: str
    occurrences: int


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    expansions: tuple[AppliedReplacement, ...]
    typo_corrections: tuple[AppliedReplacement, ...]


def _load_dictionary(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NormalizationResourceError(f"Could not load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NormalizationResourceError(f"{path} must contain a JSON object")

    normalized: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise NormalizationResourceError(f"{path} keys and values must be strings")
        key = unicodedata.normalize("NFKC", raw_key).strip().lower()
        value = unicodedata.normalize("NFKC", raw_value).strip().lower()
        if not key or not value:
            raise NormalizationResourceError(f"{path} may not contain blank keys or values")
        normalized[key] = value
    return normalized


def _replace_dictionary_phrases(
    text: str, replacements: dict[str, str]
) -> tuple[str, tuple[AppliedReplacement, ...]]:
    applied: list[AppliedReplacement] = []
    for source in sorted(replacements, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(source)}(?![a-z0-9])"
        text, count = re.subn(pattern, replacements[source], text)
        if count:
            applied.append(AppliedReplacement(source, replacements[source], count))
    return text, tuple(applied)


def normalize_text_with_trace(
    text: str, resources: NormalizationResources | None = None
) -> NormalizationResult:
    """Return normalized text plus only the dictionary changes actually applied."""

    resources = resources or NormalizationResources.load()
    normalized = unicodedata.normalize("NFKC", text).lower().strip()
    normalized = normalized.replace("’", "'").replace("–", "-").replace("—", "-")
    normalized, expansions = _replace_dictionary_phrases(normalized, resources.abbreviations)
    normalized = re.sub(r"#\s*(\d+)", r" number \1 ", normalized)
    normalized = normalized.replace("&", " and ").replace("@", " at ")

    tokens = _TOKEN_RE.findall(normalized)
    corrections: dict[tuple[str, str], int] = {}
    corrected: list[str] = []
    for token in tokens:
        replacement = resources.typo_corrections.get(token, token)
        corrected.append(replacement)
        if replacement != token:
            key = (token, replacement)
            corrections[key] = corrections.get(key, 0) + 1

    return NormalizationResult(
        text=" ".join(corrected),
        expansions=expansions,
        typo_corrections=tuple(
            AppliedReplacement(original, replacement, count)
            for (original, replacement), count in corrections.items()
        ),
    )


def normalize_text(text: str, resources: NormalizationResources | None = None) -> str:
    """Normalize text without mutating or replacing the original source value.

    Abbreviations are expanded before punctuation cleanup. Typo correction is
    exact-token and dictionary-backed; this intentionally avoids guessing at
    unfamiliar aircraft terms.
    """

    return normalize_text_with_trace(text, resources).text

