"""Read-only historical pattern explorer; never executes retrieval or an LLM."""
import json
import streamlit as st

from .pattern_build import DEFAULT_OUTPUT, SOURCE, RESOURCE_PATH, load_patterns, resource_hashes
from .pattern_graph import FRAMING, DISCLAIMER


@st.cache_resource(show_spinner=False)
def _cached(signature):
    return load_patterns()


def _load():
    try:
        paths = [SOURCE, *sorted(DEFAULT_OUTPUT.glob("*"))]
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        return _cached((signature, json.dumps(resource_hashes(), sort_keys=True)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(f"Historical pattern artifacts unavailable: {exc}")
        st.code("python -m maintenance_copilot.pattern_build")
        return [], {}


def _intro():
    st.subheader("Historical Pattern Intelligence")
    st.write(FRAMING)
    st.caption(DISCLAIMER)
    st.caption("Experimental corpus analysis · human pattern-quality review pending")


def _navigate(pattern_id):
    st.session_state["workspace_view"] = "Historical Pattern Intelligence"
    st.session_state["pattern_choice"] = pattern_id


def render_linked_patterns(answer):
    _intro()
    patterns, _ = _load()
    linked = [p for p in patterns if set(p["record_ids"]) & set(answer.cited_record_ids)] if answer.status == "answered" else []
    if not linked:
        st.info("No stable corpus-level pattern was found among the cited evidence records.")
    for p in linked:
        st.write(p["label"])
        st.caption(f"{p['size']} source records · cohesion {p['mean_similarity']:.3f} · stability {p['stability']:.3f}")
        st.button("Explore pattern " + p["pattern_id"], on_click=_navigate, args=(p["pattern_id"],), key="link_"+p["pattern_id"])


def render_patterns():
    _intro()
    patterns, records = _load()
    if not patterns:
        st.info("No stable corpus-level patterns available.")
        return
    query = st.text_input("Filter by component, symptom, or pattern label").strip().lower()
    minimum = st.number_input("Minimum cluster size", min_value=3, value=3, step=1)
    stability = st.slider("Minimum stability", 0., 1., .80, .01)
    sort = st.selectbox("Sort patterns by", ["Cluster size", "Cohesion", "Stability"])
    field = {"Cluster size":"size", "Cohesion":"mean_similarity", "Stability":"stability"}[sort]
    filtered = sorted([p for p in patterns if p["size"] >= minimum and p["stability"] >= stability
                       and query in (p["label"]+" "+" ".join(p["shared"]["components"]+p["shared"]["symptoms"])).lower()],
                      key=lambda p: (-p[field], p["pattern_id"]))
    st.caption(f"{len(filtered)} retained patterns match these filters")
    if not filtered:
        return
    st.dataframe([{k:p[k] for k in ("pattern_id", "label", "size", "mean_similarity", "minimum_similarity", "stability")} for p in filtered], hide_index=True, use_container_width=True)
    ids = [p["pattern_id"] for p in filtered]
    if st.session_state.get("pattern_choice") not in ids:
        st.session_state["pattern_choice"] = ids[0]
    selected = st.selectbox("Pattern", ids, key="pattern_choice")
    pattern = next(p for p in filtered if p["pattern_id"] == selected)
    st.subheader(pattern["label"])
    st.write(f"Source-record count: {pattern['size']} · Mean cohesion: {pattern['mean_similarity']:.3f} · Minimum cohesion: {pattern['minimum_similarity']:.3f} · Stability: {pattern['stability']:.3f}")
    st.json(pattern["shared"], expanded=False)
    st.markdown("#### Representative original PROBLEM texts")
    for sample in pattern["representatives"]:
        st.caption("Record " + sample["record_id"])
        st.text(sample["problem_original"])
    st.markdown("#### " + pattern["action_heading"])
    if pattern["action_warning"]:
        st.warning(pattern["action_warning"])
    st.caption("Action grouping rule: exact equality after existing normalization; no action is ranked or endorsed.")
    for i, variant in enumerate(pattern["action_variants"], 1):
        with st.expander(f"Variant {i} · {variant['count']} records"):
            st.caption("Record IDs: " + ", ".join(variant["record_ids"]))
            for original in variant["original_actions"]:
                st.text(original)
    record_id = st.selectbox("Record ID — original evidence", pattern["record_ids"])
    record = records[record_id]
    left, right = st.columns(2)
    with left:
        st.markdown("**Original PROBLEM**"); st.text(record.problem_original)
    with right:
        st.markdown("**Original ACTION**"); st.text(record.action_original)
    with st.expander("Fingerprint extraction audit"):
        from .pattern_fingerprints import fingerprint
        st.json(fingerprint(record).model_dump())
    st.caption(DISCLAIMER)
