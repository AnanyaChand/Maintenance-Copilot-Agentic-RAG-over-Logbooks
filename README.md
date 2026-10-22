# ✈️ Maintenance Copilot: Agentic RAG over Aviation Logbooks

A search-and-answer assistant for aircraft maintenance technicians. Describe a fault, and the copilot finds similar past logbook entries, summarises the likely causes and the actions that fixed them, and **shows the evidence**. If the evidence is weak, it says so instead of guessing.

> **Disclaimer:** This tool searches historical logbook records. It does **not** give approved maintenance procedures and does **not** diagnose aircraft. Always follow the official maintenance manual.

*Hackathon: PS 1.3 · Track 1 – Aviation Maintenance*

---

## The problem

When a technician sees a fault, the fix is often buried in years of old logbook entries. Those entries are:

- **Short and full of abbreviations** (`REPL R/H MLG STRUT`)
- **Misspelled**, so normal search misses obvious matches
- **High stakes**: in maintenance, a confident wrong answer is worse than no answer

## What it does

1. **Cleans the text**: expands abbreviations, keeps the original text untouched.
2. **Searches in three ways at once**: exact keywords, meaning, and typo-tolerant matching.
3. **Uses an AI agent with real tools** to look up records and write a short answer.
4. **Cites every claim** with a Record ID, and refuses when evidence is weak.
5. **Shows everything in a web app**: answer, source records side by side, and the agent's tool-call trace.

---

## How it works

```
Technician query
      │
      ▼
[Query rewrite]  expand shorthand
      │
      ▼
[Hybrid retrieval]  BM25 (keywords) + MiniLM/FAISS (meaning) → merged with RRF
      │
      ▼
[Rerank + safety gates]  evidence filtering, Left/Right/Fwd/Aft checks
      │
      ▼
[LangGraph agent]  search_records() → get_record() → grounded answer
      │
      ▼
[Streamlit UI]  answer | cited records | tool trace | refusal state
```

### Build phases

| # | Phase | What we did | How |
|---|-------|-------------|-----|
| 1 | Data ingestion & cleanup | Loaded the 6,169 MaintNet aviation records | Reads `IDENT`, `PROBLEM`, `ACTION`. Keeps the original text, adds a normalised copy (abbreviation dictionary, conservative typo handling). Original CSV is never modified. |
| 2 | Hybrid retrieval | Built the search engine | **BM25** (`rank_bm25`) for exact terms + **MiniLM embeddings** (`all-MiniLM-L6-v2`) in **FAISS** for meaning. Merged with **Reciprocal Rank Fusion (RRF)**. |
| 3 | Safe agent workflow | Built the reasoning flow | **LangGraph** agent with tools `search_records` and `get_record`. Every claim needs `[Record ID: X]`. Weak evidence gives *"Insufficient evidence in the available maintenance records."* |
| 4 | Streamlit web app | Built the demo UI | Chat on the left, verification panel on the right. Shows normalisation, BM25/vector/RRF scores, confidence gates, agent trace, refusal state, and a recurring-issue view. |
| 5 | Evaluation baseline | Built a benchmark *before* tuning | 75 fixed test queries measuring Hit@5, citation relevance, refusals, false confidence, latency. |
| 5 | Advanced retrieval | Stricter evidence filtering | Field-aware scoring, reranking, fault/evidence gates, safer per-record selection. |
| 6 | Risk-calibrated tiers (experimental) | Tried to recover more valid answers safely | Tier A (strong), Tier B (evidence with limits), Tier C (refuse). Threshold sweeps, uncertainty estimates, exact-match rescue, disagreement handling, human-review CSV. |

---

## Results

| Metric | Baseline | After advanced retrieval |
|---|---|---|
| Citation precision (cited records actually support the answer) | 36.23% | **93.10%** |
| False confidence on unanswerable queries | 20% | **0 / 20** |

**Phase 6 (tiered answers):** answered more supported queries, but false confidence came back (**2 / 8** unsupported queries got an answer). So we did **not** make it the default. The safer Phase 5 pipeline stays active.

---

## What makes this different

- **Refusal is a feature.** A wrong repair suggestion is worse than "I don't know".
- **Three search signals, not one.** Keywords catch part numbers, embeddings catch meaning, and character n-grams catch typos.
- **Fault fingerprinting.** The system pulls out symptom, component and side (Left / Right / Fwd / Aft) from the query. A "left engine" query will not be answered from a "right engine" record.
- **Disagreement-aware answers.** If past mechanics fixed the same fault in different ways, the copilot lists the options and warns that there is no single agreed fix.
- **Every claim is checkable.** Cited records appear next to the answer, and the tool trace shows exactly what the agent did.
- **Honest evaluation.** The test set was built before tuning, and we report the Phase 6 result that did not work out.

---

## Tech stack

Python · LangGraph · `rank_bm25` · sentence-transformers (`all-MiniLM-L6-v2`) · FAISS · cross-encoder reranker (`BAAI/bge-reranker`) · scikit-learn (char n-gram TF-IDF) · Streamlit · LLM (free tier)

## Data

[MaintNet aviation maintenance logbook](https://people.rit.edu/fa3019/MaintNet/) (University of North Dakota Aviation Program, 6,169 records). See the COLING 2020 MaintNet paper. Kaggle mirror: *Aircraft Historical Maintenance Dataset (2012–2017)*.

## Getting started

```bash
git clone <your-repo-url>
cd maintenance-copilot
pip install -r requirements.txt

# 1. Clean the data
python ingest.py --input data/maintnet.csv --output data/clean.csv

# 2. Build indexes
python build_index.py

# 3. Run the app
streamlit run app.py
```

> Replace the script names above with your actual file names.

Set your LLM API key in a `.env` file:

```
LLM_API_KEY=your_key_here
```

## Evaluation

- 75 fixed test queries (including unanswerable ones)
- Metrics: Hit@5, citation precision, correct refusals, false-confidence rate, latency

## Known limitations

- Current evaluation labels were written by an assistant. **A human reviewer must verify** whether each retrieved record really supports each result.
- Tiered (A/B/C) calibration is experimental and not the default.
- This is a strong prototype, **not** production-level safety.

## Team

*Ananya Chand,*
*Raseem Shafi,*
*Swathee K*

