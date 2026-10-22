import json

import pytest

from maintenance_copilot.normalization import (
    NormalizationResourceError,
    NormalizationResources,
    normalize_text,
)


def test_expands_abbreviations_corrects_known_typos_and_symbols():
    text = "R/H FWD BAFL SEEAL @ #2 CYL & CHKD ENG"

    assert normalize_text(text) == (
        "right hand forward baffle seal at number 2 cylinder and checked engine"
    )


def test_normalization_is_lowercase_and_whitespace_stable():
    assert normalize_text("  ENGINE\tIDLE\nOVERRIDE.  ") == "engine idle override"


def test_unknown_domain_token_is_preserved_not_guessed():
    assert normalize_text("XYZZY actuator") == "xyzzy actuator"


def test_symbol_bearing_abbreviation_expands_before_symbol_cleanup():
    assert normalize_text("R&R ENG") == "removed and replaced engine"


def test_editable_dictionary_can_be_loaded(tmp_path):
    (tmp_path / "abbreviations.json").write_text(
        json.dumps({"abc": "aircraft battery check"}), encoding="utf-8"
    )
    (tmp_path / "typo_corrections.json").write_text(
        json.dumps({"actutor": "actuator"}), encoding="utf-8"
    )

    resources = NormalizationResources.load(tmp_path)

    assert normalize_text("ABC ACTUTOR", resources) == "aircraft battery check actuator"


def test_invalid_dictionary_is_rejected(tmp_path):
    (tmp_path / "abbreviations.json").write_text("[]", encoding="utf-8")
    (tmp_path / "typo_corrections.json").write_text("{}", encoding="utf-8")

    with pytest.raises(NormalizationResourceError):
        NormalizationResources.load(tmp_path)

