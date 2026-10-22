from pathlib import Path

from streamlit.testing.v1 import AppTest

from maintenance_copilot.workbench import Catalog, ReviewStore

APP = Path(__file__).parents[1] / "review_app.py"


def widget(items, label):
    return next(item for item in items if item.label == label)


def test_review_ui_blank_defaults_autosave_reload_and_navigation(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_WORK_DIR", str(tmp_path / "human-reviews"))
    monkeypatch.delenv("REVIEW_ACCESS_TOKEN", raising=False)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    widget(app.text_input, "Reviewer name or pseudonym").set_value("UI test reviewer")
    widget(app.checkbox, "I am entering my own human judgments").check()
    widget(app.checkbox, "Focused review pack only").uncheck()
    app.run()
    assert not app.exception
    widget(app.selectbox, "Query ID").select("S01").run()
    assert widget(app.selectbox, "Relevance").value is None
    assert widget(app.selectbox, "Answer grounding").value is None
    widget(app.selectbox, "Relevance").select("relevant").run()
    assert not app.exception
    store = ReviewStore(tmp_path / "human-reviews", Catalog())
    saved, revision = store.load("UI test reviewer", "S01")
    assert revision == 1
    assert saved.records["100001"].relevance == "relevant"
    assert saved.answer_grounding is None
    widget(app.button, "Next").click().run()
    assert widget(app.selectbox, "Query ID").value != "S01"
    widget(app.selectbox, "Query ID").select("S01").run()
    assert widget(app.selectbox, "Relevance").value == "relevant"
    assert not app.exception


def test_review_ui_access_token_blocks_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_WORK_DIR", str(tmp_path / "blocked"))
    monkeypatch.setenv("REVIEW_ACCESS_TOKEN", "test-only-secret")
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert not (tmp_path / "blocked").exists()
    assert not any(s.label == "Query ID" for s in app.selectbox)
