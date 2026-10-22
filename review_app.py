"""Separate local-only Human Review Workbench. Never changes production retrieval."""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path

import streamlit as st

from maintenance_copilot.workbench import (
    CASE_FIELDS, RECORD_FIELDS, CaseReview, Catalog, ReviewStore, DEFAULT_REVIEW_DIR,
    filter_cases, focused_pack,
)
from maintenance_copilot.workbench_reports import report, agreement_report

st.set_page_config(page_title="Human Review Workbench | Maintenance Copilot", page_icon="✈", layout="wide")


@st.cache_resource
def catalog():
    return Catalog()


def protect():
    token = os.getenv("REVIEW_ACCESS_TOKEN")
    if token:
        entered = st.text_input("Workbench access token", type="password", key="access_token")
        if not hmac.compare_digest(entered, token):
            st.info("Enter the access token to load review evidence.")
            st.stop()
    else:
        st.caption("Local-only review workspace · No authentication configured. Bind to 127.0.0.1; do not expose publicly.")


def autosave(store, field=None, record_id=None, key=None):
    draft = dict(st.session_state["draft"])
    if field:
        value = st.session_state[key]
        if field == "suggested_expected_record_ids":
            value = [x.strip() for x in value.replace("\n", ",").split(",") if x.strip()]
        if field == "reviewed_at":
            value = value or None
        if record_id:
            draft["records"] = dict(draft.get("records", {}))
            labels = dict(draft["records"].get(record_id, {}))
            labels[field] = value
            draft["records"][record_id] = labels
        else:
            draft[field] = value
    st.session_state["draft"] = draft
    try:
        review = CaseReview.model_validate(draft)
        revision = store.save(review, st.session_state["revision"])
        saved, _ = store.load(review.reviewer_name, review.case_id)
        st.session_state["draft"] = saved.model_dump(mode="json")
        st.session_state["revision"] = revision
        st.session_state["save_message"] = f"Saved revision {revision} · {saved.reviewed_at.isoformat()}"
        st.session_state.pop("save_error", None)
    except Exception as exc:
        st.session_state["save_error"] = str(exc)


def reload_case():
    st.session_state.pop("active_review", None)
    st.session_state.pop("save_error", None)
    st.session_state.pop("save_message", None)


def select_label(store, field, value, prefix, *, rid=None, options=None, help=None):
    key = f"{prefix}:{rid or 'case'}:{field}"
    options = options or (RECORD_FIELDS if rid else CASE_FIELDS)[field]
    st.selectbox(field.replace("_", " ").capitalize(), options,
                 index=options.index(value) if value in options else None,
                 placeholder="Not reviewed — choose a judgment", key=key, help=help,
                 on_change=autosave, args=(store, field, rid, key))


def text_field(store, field, value, prefix, *, help=None):
    key = f"{prefix}:{field}"
    st.text_area(field.replace("_", " ").capitalize(), value=value or "", key=key,
                 on_change=autosave, args=(store, field, None, key), help=help, height=75)


def context_panel(cat, case):
    answer = case["answer"]
    st.subheader(f"{case['id']} · {case['tier'].replace('TIER_', 'Tier ').replace('_', ' ').title()}")
    st.caption(f"{case['split']} · {case['category']} · {case['status']} · archived selective experiment, NOT the production default")
    left, right = st.columns(2)
    with left:
        st.caption("ORIGINAL QUERY")
        st.write(answer["query"])
    with right:
        st.caption("NORMALIZED QUERY USED IN RETRIEVAL")
        st.write(answer["normalized_query"])
    with st.expander("Query changes and benchmark variants"):
        st.json(answer["query_rewrite"])
        st.caption("No additional query variants were logged as executed. Related benchmark cases below were separate queries, not tool calls for this case.")
        family = cat.labels[case["id"]]["family"]
        st.dataframe([{"case": c["id"], "query": c["query"]} for c in cat.labels.values() if c["family"] == family], hide_index=True)
    with st.expander("Provisional expected record IDs — reveal after independent judgment"):
        st.warning("Assistant-authored, non-exhaustive expectations. Do not treat these as human ground truth.")
        st.write(", ".join(case["expected_ids"]) or "No provisional positive IDs")
    with st.container(border=True):
        st.caption("SYSTEM ANSWER — OBJECT OF REVIEW, NOT MAINTENANCE ADVICE")
        st.text(answer["answer_text"])
    st.metric("System heuristic confidence", f"{answer['confidence']['score']:.3f}", help="Not a calibrated probability of correctness.")
    with st.expander("Evidence gates, limitations and recorded tool trace"):
        for gate in answer["confidence"]["explanation"]:
            st.write(gate)
        if answer.get("selective"):
            st.json(answer["selective"], expanded=False)
        st.dataframe([{"tool": e["tool_name"], "arguments": json.dumps(e["arguments"]),
                       "result_ids": ", ".join(e["result_record_ids"]), "status": e["status"],
                       "duration_ms": round(e["duration_ms"], 2), "summary": e["result_summary"]}
                      for e in answer["tool_trace"]], hide_index=True)
        st.caption("Development traces replayed cached candidate features during calibration; held-out traces came from live inference. Nothing is rerun by this workbench.")


def record_panel(cat, case, rid):
    info = cat.evidence(case["id"], rid)
    record, hit, feature = info["record"], info["hit"], info["feature"]
    st.caption(f"{info['role']} · Record {rid}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original PROBLEM**")
        st.write(record.problem_original)
    with right:
        st.markdown("**Original ACTION**")
        st.write(record.action_original)
    with st.expander("Normalized source text"):
        st.write({"problem_normalized": record.problem_normalized, "action_normalized": record.action_normalized})
    st.caption("Score provenance: " + info["score_origin"] + ". Missing values were not retained; they are not zero.")
    rows = []
    for label, score, rank in (("BM25", "bm25_score", "bm25_rank"), ("Dense cosine", "vector_similarity", "vector_rank"),
                              ("Character", "character_score", "character_rank"), ("RRF", "rrf_score", "rrf_rank"),
                              ("Field reranker", "final_score", "final_rank")):
        rows.append({"signal": label, "score": str(hit.get(score)) if hit.get(score) is not None else "Unavailable",
                     "rank": str(hit.get(rank)) if hit.get(rank) is not None else "Unavailable"})
    ce_rank = None
    if feature:
        ordered = sorted(cat.report["development_features"][case["id"]]["candidates"], key=lambda c: (-c["cross_logit"], c["record_id"]))
        ce_rank = next(i for i, c in enumerate(ordered, 1) if c["record_id"] == rid)
    rows.append({"signal": "Cross-encoder logit (12-candidate pool)", "score": str(feature.get("cross_logit", "Unavailable")),
                 "rank": str(ce_rank) if ce_rank else "Unavailable"})
    rows.append({"signal": "Fault fingerprint consistency", "score": str(feature.get("fingerprint", "Unavailable")), "rank": "Not applicable"})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if hit.get("evidence_gate_reasons"):
        st.caption("Archived field-gate warnings (not a reconstructed selective decision): " + "; ".join(hit["evidence_gate_reasons"]))


def main():
    st.markdown("# Human Review Workbench")
    st.caption("Maintenance Copilot · Independent human judgment over frozen historical evidence")
    protect()
    cat = catalog()
    store = ReviewStore(Path(os.getenv("REVIEW_WORK_DIR", str(DEFAULT_REVIEW_DIR))), cat)
    pack = focused_pack(cat)
    with st.sidebar:
        st.header("Review session")
        reviewer = st.text_input("Reviewer name or pseudonym", key="reviewer", disabled=bool(st.session_state.get("save_error"))).strip()
        attested = st.checkbox("I am entering my own human judgments", key="human_attestation")
        st.caption("Names are self-declared, not verified identities. Do not share a reviewer name across people.")
        pack_only = st.checkbox("Focused review pack only", value=True)
        st.caption(f"{len(pack['cases'])} unique pack cases; action-divergence availability is limited.")
        status = st.selectbox("System result", ["all", "answered", "insufficient_evidence"])
        tier = st.selectbox("Evidence tier", ["all", "TIER_A_STRONG_EVIDENCE", "TIER_B_SUPPORTED_WITH_LIMITATIONS", "TIER_C_INSUFFICIENT_EVIDENCE"])
        split = st.selectbox("Benchmark split", ["all", "development", "held_out"])
        flags = st.multiselect("Queue filters", ["unreviewed", "high_confidence", "suspected_false_confidence", "action_divergence"])
        search = st.text_input("Search query text or record ID")
        st.caption("High confidence ≥0.80. Suspected false confidence is a provisional screening flag, not a human judgment.")
    review_tab, guide_tab, metrics_tab, exchange_tab = st.tabs(["Review cases", "Annotation guide", "Agreement & metrics", "Import / export"])
    with guide_tab:
        st.markdown((Path(__file__).parent / "evaluation/ANNOTATION_GUIDE.md").read_text())
    with metrics_tab:
        current_report = report(cat, store)
        st.subheader("Human-reviewed labels")
        if current_report["human_reviewed_labels"] is None:
            st.info("No human reviews saved. Human metrics are unmeasured, not 100%.")
        else:
            st.json(current_report["human_reviewed_labels"], expanded=False)
        st.subheader("Inter-reviewer agreement")
        st.json(current_report["agreement"], expanded=False)
        queue = current_report["agreement"]["disagreement_queue"]
        if queue:
            st.dataframe(queue, hide_index=True)
        st.subheader("Provisional automated labels — separate reference")
        st.json(current_report["provisional_automated_labels"], expanded=False)
        st.download_button("Download labeled metrics report", json.dumps(current_report, indent=2), "human_metrics.json", "application/json")
    with exchange_tab:
        st.caption(f"Original (read-only): {cat.original_csv}")
        st.caption(f"Human work: {store.directory} · SQLite is authoritative; versioned CSV snapshots follow every save.")
        st.download_button("Export all human reviews (CSV)", store.export_csv(), "human_review_v1.csv", "text/csv")
        st.download_button("Export focused pack (JSON)", json.dumps(pack, indent=2), "focused_review_pack.json", "application/json")
        st.download_button("Export original evidence pool (read-only CSV)", cat.original_csv.read_bytes(), "provisional_review_pool.csv", "text/csv")
        st.caption("The provisional evidence CSV is not a human-review import. Complete labels in this workbench, then use its exported CSV for transfer or further edits.")
        uploaded = st.file_uploader("Import a workbench-reviewed CSV", type="csv")
        confirmed = st.checkbox("I confirm this import contains actual human judgments and I am authorized to import these reviewers' work")
        if st.button("Validate and import atomically", disabled=not (uploaded and reviewer and attested and confirmed)):
            try:
                saved = store.import_csv(uploaded.getvalue().decode("utf-8-sig"), actor=reviewer)
                reload_case()
                st.success(f"Imported {len(saved)} case reviews. Unchanged duplicate exports were skipped.")
            except Exception as exc:
                st.error(str(exc))
        st.markdown("#### Reviewer-specific audit trail")
        audit = store.audit(reviewer or None)
        st.download_button("Export audit trail (JSON)", json.dumps(audit, indent=2), "review_audit.json", "application/json")
        if audit:
            st.dataframe([{k: e[k] for k in ("event_id", "actor", "case_id", "kind", "server_at")} for e in audit[-30:]], hide_index=True)
    with review_tab:
        if not reviewer or not attested:
            st.info("Enter your reviewer name and confirm human authorship to begin. No labels are preselected or saved automatically from model output.")
            return
        reviews = {r.case_id: r for r, _ in store.all() if r.reviewer_name == reviewer}
        pool = list(pack["cases"]) if pack_only else list(cat.cases)
        done = sum(cat.complete(reviews.get(cid)) for cid in pool)
        st.progress(done / len(pool), text=f"{done}/{len(pool)} cases complete for {reviewer} · unsure is explicit but excluded from agreement/metric denominators")
        queue = filter_cases(cat, reviews, status=status, tier=tier, split=split, flags=flags, search=search,
                             pack=set(pack["cases"]) if pack_only else None)
        if st.session_state.get("save_error") and st.session_state.get("active_review", (None,))[0] == reviewer:
            queue = [st.session_state["active_review"][1]]
            st.warning("Queue changes are deferred until the unsaved/conflicting edit is corrected or reloaded.")
        if not queue:
            st.info("No cases match this queue. Adjust filters or switch off the focused pack.")
            return
        if st.session_state.get("case_choice") not in queue:
            st.session_state["case_choice"] = queue[0]
        def move(offset):
            index = queue.index(st.session_state["case_choice"])
            st.session_state["case_choice"] = queue[max(0, min(len(queue)-1, index+offset))]
        def next_unreviewed():
            index = queue.index(st.session_state["case_choice"])
            cyclic = queue[index+1:] + queue[:index+1]
            st.session_state["case_choice"] = next((cid for cid in cyclic if not cat.complete(reviews.get(cid))), queue[index])
        c1, c2, c3, c4 = st.columns([1, 1, 2, 3])
        blocked = bool(st.session_state.get("save_error"))
        c1.button("Previous", on_click=move, args=(-1,), disabled=blocked)
        c2.button("Next", on_click=move, args=(1,), disabled=blocked)
        c3.button("Next unreviewed case", on_click=next_unreviewed, disabled=blocked)
        cid = c4.selectbox("Query ID", queue, key="case_choice", disabled=blocked)
        active = (reviewer, cid)
        if st.session_state.get("active_review") != active:
            st.session_state.pop("save_message", None)
            saved, revision = store.load(reviewer, cid)
            draft = saved or CaseReview(case_id=cid, reviewer_name=reviewer)
            st.session_state.update(active_review=active, draft=draft.model_dump(mode="json"), revision=revision,
                                    generation=st.session_state.get("generation", 0)+1)
        prefix = f"review:{st.session_state['generation']}:{reviewer}:{cid}"
        case = cat.cases[cid]
        if st.session_state.get("save_error"):
            st.error("Save blocked: " + st.session_state["save_error"] + ". Correct the edit or reload the saved review; no conflicting annotation was overwritten.")
        elif st.session_state.get("save_message"):
            st.success(st.session_state["save_message"])
        actions = st.columns(2)
        actions[0].button("Save review now", on_click=autosave, args=(store,), type="primary")
        actions[1].button("Reload saved review (discard unsaved edits)", on_click=reload_case)
        st.caption("Autosave runs after each committed control change. Text fields save when you leave the field or submit the edit, not on every keystroke.")
        others = [r.reviewer_name for r, _ in store.all() if r.case_id == cid and r.reviewer_name != reviewer]
        if others:
            st.warning("Independent reviews already exist from: " + ", ".join(others) + ". Yours stays separate; disagreements are queued, never overwritten.")
        context_panel(cat, case)
        st.divider()
        st.subheader("System-level human judgment")
        draft = st.session_state["draft"]
        columns = st.columns(3)
        for col, field in zip(columns, list(CASE_FIELDS)[:3]):
            with col:
                select_label(store, field, draft.get(field), prefix)
        with st.expander("Optional audit needed for refusal recall and citation coverage"):
            select_label(store, "evidence_sufficiency", draft.get("evidence_sufficiency"), prefix,
                         help="Is there enough historical evidence to provide a bounded answer? Judge independently of whether this system answered. Without this, false confidence on answered cases cannot be measured.")
            for field in ("meaningful_claim_count", "claims_with_supported_citations"):
                key = f"{prefix}:{field}"
                st.number_input(field.replace("_", " ").capitalize(), min_value=0, max_value=10000,
                    value=draft.get(field), step=1, key=key, on_change=autosave, args=(store, field, None, key))
        text_field(store, "reviewer_notes", draft.get("reviewer_notes"), prefix)
        text_field(store, "suggested_expected_record_ids", ", ".join(draft.get("suggested_expected_record_ids", [])), prefix,
                   help="Comma-separated canonical record IDs. Suggestions do not change the frozen benchmark.")
        with st.expander("Optional review date"):
            text_field(store, "reviewed_at", draft.get("reviewed_at"), prefix,
                       help="ISO 8601 date/time; blank uses first save time. Every edit also has an immutable server audit timestamp.")
        st.divider()
        st.subheader("Record evidence and relevance")
        if case["pool"]:
            rid = st.selectbox("Record to review", case["pool"], key=f"{prefix}:record_choice",
                               format_func=lambda r: f"{r} · {cat.evidence(cid, r)['role']}")
            record_panel(cat, case, rid)
            judgment = draft.get("records", {}).get(rid, {})
            cols = st.columns(3)
            for col, field in zip(cols, RECORD_FIELDS):
                with col:
                    select_label(store, field, judgment.get(field), prefix, rid=rid)
            st.caption(f"{sum(all(v.get(k) is not None for k in RECORD_FIELDS) for v in draft.get('records', {}).values())}/{len(case['pool'])} records have all three labels.")
        st.caption("Historical logbook evidence is not an approved repair procedure. No production model, threshold, or benchmark label is changed by a review.")


if __name__ == "__main__":
    main()
