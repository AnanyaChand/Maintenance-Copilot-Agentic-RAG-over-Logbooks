# Maintenance Copilot

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

## Phase 9: Historical Pattern Intelligence

Repeated historical issue pattern across the MaintNet corpus

Patterns are similarity-based groupings of historical logbook text.
They do not establish aircraft-specific recurrence, cause, or approved maintenance action.

In the main app select **Workspace → Historical Pattern Intelligence** for the
problem-only graph explorer. Normal fault answers link retained patterns using
cited record IDs. Retrieval defaults and agent safety gates are unchanged.

See [method, measured results, review artifact and reproduction commands](docs/historical-patterns.md).
The initial build retains 301 patterns; human quality review is pending.

## Phase 8: optional LLM tool calling

The default remains the deterministic extractive agent, with no API key required.
An explicitly configured `COPILOT_MODE=llm` and `LLM_ENABLED=true` enable an
experimental OpenAI-compatible tool-calling provider. Local retrieval gates run
before API access. The model must then request real search and fetch calls; a
strict JSON contract and deterministic whole-field verifier validate its output.
Errors, timeouts and rejected output fall back to the existing deterministic agent.

See [Phase 8 setup, evaluation and limitations](evaluation/PHASE8.md). Configure
the blank LLM settings in `.env.example` in your private `.env`; do not commit keys.
The UI shows the actual outcome and safe fallback reason. No model name is chosen
for you. This phase adds no new runtime dependencies.

```bash
# Offline protocol test over the frozen suite; no frontier model/API calls.
.venv/bin/python -m maintenance_copilot.llm_evaluation --provider mock \
  --output evaluation/results/my-phase8-mock.json
```

