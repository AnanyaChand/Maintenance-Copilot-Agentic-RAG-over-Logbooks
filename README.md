# ✈️ Maintenance Copilot: Agentic RAG over Aviation Logbooks

A search-and-answer assistant for aircraft maintenance technicians. Describe a fault, and the copilot finds similar past logbook entries, summarises the likely causes and the actions that fixed them, and shows the evidence. If the evidence is weak, it says so instead of guessing.

Disclaimer: This tool searches historical logbook records. It does not give approved maintenance procedures and does not diagnose aircraft. Always follow the official maintenance manual.

Hackathon: PS 1.3 · Track 1 – Aviation Maintenance

The problem

When a technician sees a fault, the fix is often buried in years of old logbook entries. Those entries are:

Short and full of abbreviations (REPL R/H MLG STRUT)
Misspelled, so normal search misses obvious matches
High stakes: in maintenance, a confident wrong answer is worse than no answer
What it does
Cleans the text: expands abbreviations, keeps the original text untouched.
Searches in three ways at once: exact keywords, meaning, and typo-tolerant matching.
Uses an AI agent with real tools to look up records and write a short answer.
Cites every claim with a Record ID, and refuses when evidence is weak.
Shows everything in a web app: answer, source records side by side, and the agent's tool-call trace.
Phase 1 provides validated ingestion and normalization for the MaintNet aviation maintenance CSV. It references the source dataset through `DATASET_PATH`; it does not copy or alter the source file.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

## Ingest

```bash
maintenance-copilot ingest
```

The command reads `DATASET_PATH`, validates and normalizes the records, writes JSON Lines to `data/processed/records.jsonl`, and writes an audit report beside it. An explicit path can be supplied without changing the environment:

```bash
maintenance-copilot ingest --dataset /absolute/path/to/file.csv --output /absolute/path/to/records.jsonl
```

The editable normalization dictionaries live in `src/maintenance_copilot/resources/`. Corrections are exact and dictionary-backed so technical terms are not guessed.

## Test

```bash
pytest
```

## Build and search the hybrid index

The index combines weighted normalized/original BM25 tokens with normalized
`all-MiniLM-L6-v2` embeddings in an exact FAISS `IndexFlatIP`. Both retrieval
channels search 50 candidates and are fused with Reciprocal Rank Fusion.

```bash
maintenance-copilot build-index
maintenance-copilot search "R/H FWD baffle seal loose" --top-k 5
```

The search response includes the query rewrite audit, raw BM25 and cosine
scores, ranks, RRF score, IDF-weighted token coverage, and the confidence gates.
Unsupported queries return the exact refusal message required by the project.

## Evidence-grounded agent (Phase 3)

```bash
maintenance-copilot answer "engine idle override killed engine"
```

`maintenance_copilot.agent.answer_query()` returns a structured answer with
citations, original evidence, retrieval confidence, disclaimer, and timed tool
events. It uses the existing search gate, then fetches up to three records with
`get_record`. Refusal stops before fetching or synthesis. The default provider
requires no API key and quotes complete source fields. Historical actions are
not assumed successful or approved procedures. Causes remain unasserted.

Providers implement `SynthesisProvider`; every output passes the same strict
source and citation checks. This phase includes the deterministic provider only.
Tool failures raise `AgentExecutionError` with a `tool_trace`; grounding failures
raise `GroundingError`. Neither produces a maintenance answer. The confidence
score remains a retrieval heuristic, not a probability of diagnosis correctness.

## Streamlit demo (Phase 4)

After installation and index construction, start the local app from the project root:

```bash
streamlit run streamlit_app.py
```

The interface runs every submitted query through `answer_query`, showing the
actual search/fetch trace. It displays the original and normalized query,
applied expansions, side-by-side cited records with BM25/vector/RRF scores,
the refusal gate, and corpus-level repeated issue patterns. The detector
counts exact normalized issue text across MaintNet records that match a cited
record. Because the dataset has no aircraft ID or dates, it cannot establish
repeat defects on a particular aircraft or a time sequence.

The local source CSV is never accessed by the UI. The app needs the processed
JSONL, retrieval artifacts, and a cached MiniLM model. The sidebar explains
how to create missing data and indexes. A hosted deployment would need those
artifacts and the model available in its environment.

## Evaluation-first intelligence upgrade

The frozen 75-case benchmark, metric definitions, review procedure and actual local
baseline are in [evaluation/README.md](evaluation/README.md) and
[evaluation/RESULTS.md](evaluation/RESULTS.md). Queries are assistant-authored from
real MaintNet records and explicitly await human review; they are not represented
as human-written. Production retrieval and safety gates remain unchanged.

```bash
.venv/bin/python -m maintenance_copilot.evaluation run \
  --output evaluation/results/my-baseline.json --repeats 3
```

The command runs the actual agent plus ranking-only BM25/vector ablations, records
real traces, checks dataset integrity, and refuses to overwrite existing results.
It requires the existing canonical data, built index and locally cached model;
no API key or new model download is needed.

## Experimental field-aware retrieval

The baseline is still the default. Opt-in `field-aware` and `field-gated` variants
add separately weighted problem/action fields, exact technical-identifier signals,
and (for `field-gated`) deterministic reranking and per-record citation eligibility.
See [the experiment protocol](evaluation/ADVANCED_RETRIEVAL.md) for weights,
limitations, benchmark commands and rollback. The UI shows the active variant and
field-level evidence diagnostics; every search still goes through the real agent.

```bash
RETRIEVAL_VARIANT=field-gated .venv/bin/streamlit run streamlit_app.py
```

Restart with `RETRIEVAL_VARIANT=baseline` (or unset it) to use the original pipeline.

## Phase 6: selective calibration (not promoted)

[Phase 6 report](evaluation/PHASE6.md) documents evidence tiers, a local problem-only
cross-encoder, a 64-combination development sweep, family-separated held-out results,
empirical uncertainty bands, bootstrap intervals, curves and a CSV human-review workflow.
The experiment recovered coverage but failed safety acceptance. Baseline routing and
the UI remain unchanged; the selective backend is evaluation-only pending successful
calibration and human review.

```bash
.venv/bin/python -m maintenance_copilot.calibration --output evaluation/phase6-repeat
```

## Phase 7: human evidence review

The independent, offline review workbench reads archived results; it does not
change production retrieval or generate human labels. See
[the handoff guide](evaluation/PHASE7.md) and
[annotation guidelines](evaluation/ANNOTATION_GUIDE.md).

```bash
.venv/bin/streamlit run review_app.py --server.address 127.0.0.1 --server.port 8502
```

Human judgments start blank and are autosaved separately under
`evaluation/reviews/`, with revision checks, audit history and versioned exports.
The focused pack contains 57 unique cases, including all seven available
action-divergence cases. Human metrics remain unavailable until humans annotate.

```bash
.venv/bin/python -m maintenance_copilot.workbench_reports report
```
