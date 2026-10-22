import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from maintenance_copilot.models import CanonicalRecord
from maintenance_copilot.normalization import normalize_text
from maintenance_copilot.pattern_fingerprints import fingerprint, conflicts, digest
from maintenance_copilot.pattern_graph import analyze, PatternConfig, ACTION_HEADING, ACTION_WARNING, FRAMING
from maintenance_copilot.pattern_build import ROOT, SOURCE, DEFAULT_OUTPUT, load_patterns, VERSION, resource_hashes


def record(i, problem="engine oil leaking", action="inspected engine"):
    return CanonicalRecord(record_id=str(i), problem_original=problem, action_original=action,
        problem_normalized=normalize_text(problem), action_normalized=normalize_text(action), source_row=i+2)


def run(problems, vectors=None, actions=None):
    records = [record(i, p, actions[i] if actions else "inspected") for i,p in enumerate(problems)]
    vectors = np.tile([1.,0.], (len(records),1)) if vectors is None else vectors
    return analyze(records, vectors)


def test_identical_and_supported_nonidentical_cluster():
    for problems in (["engine oil leaking"]*3, ["engine oil leaking", "oil leaking near engine", "engine leaking oil"]):
        result = run(problems)
        assert len(result["patterns"]) == 1
        p = result["patterns"][0]
        assert p["size"] == 3 and p["stability"] == 1
        assert p["label"] == "Repeated engine leaking reports"
        assert set(p["record_ids"]) == {"0","1","2"}
        assert p["action_heading"] == ACTION_HEADING


@pytest.mark.parametrize("a,b", [("LH","RH"),("FWD","AFT"),("upper","lower"),("#2","#3")])
def test_conflicts_never_form_visible_cluster(a,b):
    result = run([f"{a} engine oil leaking", f"{b} engine oil leaking", "engine oil leaking"])
    assert not result["patterns"]
    assert conflicts(fingerprint(record(0,f"{a} engine oil leaking")), fingerprint(record(1,f"{b} engine oil leaking")))


def test_action_similarity_cannot_create_edges():
    assert not run(["engine leaking", "window cracked", "battery missing"])["edges"]
    a = run(["engine leaking"]*3)
    b = run(["engine leaking"]*3, actions=["replace", "inspect", "no action"])
    assert a["edges"] == b["edges"]
    assert a["patterns"][0]["record_ids"] == b["patterns"][0]["record_ids"]
    assert b["patterns"][0]["action_warning"] == ACTION_WARNING
    assert len(b["patterns"][0]["action_variants"]) == 3


def test_size_and_stability_gates():
    assert not run(["engine leaking"]*2)["patterns"]
    # Pairwise similarity .85: eligible nominally, dissolves at .86 perturbation.
    matrix = np.full((3,3), .85); np.fill_diagonal(matrix, 1.)
    vectors = np.linalg.cholesky(matrix)
    records = [record(i,p) for i,p in enumerate(["engine leaking", "engine oil leaking", "oil leaking engine"])]
    result = analyze(records, vectors, PatternConfig(mean_cohesion=.84))
    assert not result["patterns"]
    assert "unstable membership" in result["audits"]["complete_link"][0]["exclusion_reasons"]


def test_fingerprint_audit_and_original_tokens():
    fp = fingerprint(record(0,"L/H eng AB-123 leaking"))
    assert "left" in fp.locations and "AB-123" in fp.original_technical_tokens
    assert fp.extraction_audit and fp.resource_checksum and fp.abbreviation_expanded_terms


def test_deterministic_ids_and_shared_label():
    a = run(["engine leaking"]*3)
    b = run(["engine leaking"]*3)
    assert a == b
    p = a["patterns"][0]
    assert p["label"] == "Repeated engine leaking reports"
    assert "engine" in p["shared"]["components"] and "leaking" in p["shared"]["symptoms"]


def test_missing_location_bridge_is_not_a_visible_conflicting_pattern():
    result = run(["LH engine leaking"]*3 + ["engine leaking"]*3 + ["RH engine leaking"]*3)
    for p in result["patterns"]:
        assert not p["conflicts"]
        assert not ({"0", "6"} <= set(p["record_ids"]))
    assert any("critical fingerprint conflict" in p["exclusion_reasons"] for p in result["audits"]["connected_components"])


def test_load_fails_closed_on_checksum_or_source_changes(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(record(0).model_dump_json()+"\n")
    output = tmp_path / "patterns"; output.mkdir()
    data = output / "patterns.json"; data.write_text("[]")
    manifest = dict(version=VERSION, source_sha256=digest(source), resources=resource_hashes(), checksums={"patterns.json":digest(data)})
    (output / "manifest.json").write_text(json.dumps(manifest))
    assert load_patterns(output, source)[0] == []
    data.write_text("[{}]")
    with pytest.raises(ValueError, match="checksum"):
        load_patterns(output, source)
    source.write_text(record(1).model_dump_json()+"\n")
    with pytest.raises(ValueError, match="stale"):
        load_patterns(output, source)


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="real pattern artifacts not built")
def test_saved_ids_integrity_and_main_ui_drilldown():
    patterns, records = load_patterns()
    assert patterns
    assert all(set(p["record_ids"]) <= records.keys() for p in patterns)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio(key="workspace_view").set_value("Historical Pattern Intelligence").run()
    assert not app.exception
    assert any(FRAMING == x.value for x in app.markdown)
    selection = next(s for s in app.selectbox if s.label == "Record ID — original evidence")
    selected = selection.options[-1]
    selection.set_value(selected).run()
    assert not app.exception
    assert records[selected].problem_original in [t.value for t in app.text]
    assert records[selected].action_original in [t.value for t in app.text]
    app.text_input[0].set_value("zzzz unmatched pattern").run()
    assert not app.exception
    assert any("0 retained" in x.value for x in app.caption)


@pytest.mark.parametrize("relative,expected", [
    ("src/maintenance_copilot/retrieval.py", "11b734642f4afedd51ddf14b361eab29f13b77ce9adb1a1b74cba9dfb63d8676"),
    ("src/maintenance_copilot/confidence.py", "bc26e767537060b67e7ef227d588239a857cf7602d7d64e71b5b254e82223156"),
    ("src/maintenance_copilot/agent.py", "235d14c392f11ca9f0b4a4ed5964551404e4d021127beeac37c922e7a5b26668"),
    ("src/maintenance_copilot/llm_agent.py", "91eb9d7c68feda1fa313266a1e6f27ff822d3df056b9c907569df1efd67a5eb3"),
    ("src/maintenance_copilot/indexing.py", "c204c912351cc1b421a1a70a91e70469ba019c034ea500864360cd6d0f20df0a"),
    ("src/maintenance_copilot/normalization.py", "684069d204f5009e72080ee2ab36450d2252f644324d3a46a9f141b90a4da8eb"),
    ("src/maintenance_copilot/llm_config.py", "731b0ca00251203310536d2d3718f31ed0a32b5992a393e5df15f7c32139b408"),
    ("src/maintenance_copilot/llm_grounding.py", "5c8f04c360ad6a5025e470bbaa947e2bc68f6a69746d03fd6923c4cf600b8189"),
    ("artifacts/bm25.pkl", "1db3a9c30a93f6299e2597e421e20ce6694ac4b4e2c852aac9cb30b793fdddfb"),
    ("artifacts/faiss.index", "7427afde4ae6cea13ec31ebea75e8a2f08325261ba8113efec6f4f5a84ada174"),
    ("artifacts/embeddings.npy", "0743f86e04fd39837a6ae84420c088433aa66239062406a5659440ff46e9f857"),
    ("artifacts/manifest.json", "8a355aec4fe41cee0fcc69f2b33a9a9397940f28e84872a40f405cc161509782"),
    ("evaluation/benchmark.v1.json", "7bc9706f498aedaf6356f5802d3d7f7cd2977e228e464f091080710675f7e1df"),
])
def test_baseline_checksums_preserved(relative, expected):
    path = ROOT / relative
    if not path.exists(): pytest.skip("optional local artifact")
    assert digest(path) == expected


def test_raw_source_unchanged():
    raw = ROOT.parent / "Aircraft_Annotation_DataFile.csv"
    if not raw.exists(): pytest.skip("raw dataset not installed")
    assert digest(raw) == "a70daa27ba9c44a68617d69be1352be7fde003145ca142cbe0b7b51e197c6c3a"
