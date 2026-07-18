# Progress Tracker — Motif Offline Multimodal RAG

> **Purpose:** Living document. Updated by the implementer as work progresses.  
> **Do not batch-update this file.** Update it in real time as each task is completed.

---

## Phase Status Overview

| Phase | Name | Status | Started | Completed |
|---|---|---|---|---|
| **0** | Infrastructure (REPL, installer, packaging) | 🔲 Not started | — | — |
| **1** | Foundation (Text RAG MVP) | 🔲 Not started | — | — |
| **2** | Quality (85% faithfulness) | 🔲 Not started | — | — |
| **3** | Multimodal Ingestion | 🔲 Not started | — | — |
| **4** | Production Hardening | 🔲 Not started | — | — |
| **5** | Optional Enhancements | 🔲 Not started | — | — |

Legend: 🔲 Not started | 🔄 In progress | ✅ Done | ❌ Blocked

---

## Phase 0 — Infrastructure

**Goal:** `motif` is an installable command on all platforms. The REPL launches with a working welcome screen. Slash commands are stubbed. Pipeline does not exist yet.

### Tasks

- 🔲 `pyproject.toml` — package metadata, `motif = "rag.cli:main"` entry point, pinned deps
- 🔲 `install.sh` — Linux/macOS bootstrap: check/install uv, install Motif, detect CUDA
- 🔲 `install.ps1` — Windows PowerShell bootstrap: same logic
- 🔲 `config.template.toml` — fully commented config with T1/T2/T3 variants shown
- 🔲 `rag/config.py` — config dataclasses, `detect_hardware_tier()`
- 🔲 `rag/session.py` — Session class: history list, `add_turn()`, `save()`, `load()`, `clear()`, `new()`, rolling window trim
- 🔲 `cli.py` — prompt_toolkit REPL: welcome screen (Rich panel), REPL loop, slash command router, session load/save
- 🔲 `rag/commands/__init__.py` — command registry
- 🔲 `rag/commands/help.py` — `/help` (lists all commands)
- 🔲 `rag/commands/clear.py` — `/clear`, `/new`
- 🔲 `rag/commands/status.py` — `/status` stub (shows "no index yet" until Phase 1)
- 🔲 `rag/commands/ingest.py` — `/ingest` stub
- 🔲 `rag/commands/remove.py` — `/remove` stub
- 🔲 `rag/commands/sync.py` — `/sync` stub
- 🔲 `rag/commands/setup.py` — `/setup` stub (calls `setup_models.py`)
- 🔲 `setup_models.py` — model download with tier detection and progress bars
- 🔲 `rag/models/__init__.py`
- 🔲 `rag/models/embedder.py` — nomic-embed ONNX wrapper class `Embedder`
- 🔲 `rag/models/reranker.py` — cross-encoder ONNX wrapper class `Reranker`
- 🔲 `rag/types.py` — **all shared dataclasses**: `Chunk`, `ScoredPassage`, `Citation`, `AnswerResult`, `IngestResult`, `SyncResult`
- 🔲 `rag/ingestion/__init__.py` — **public API stubs** (not implemented): `ingest_path()`, `remove_document()`, `sync_directory()` with correct type signatures using types from `rag.types`
- 🔲 `tests/conftest.py` — shared pytest fixtures: `tmp_db_root`, `sample_pdf`, `sample_md`, `minimal_config`

### Phase 0 Acceptance Checkpoint

- 🔲 `motif` command exists on PATH after install script runs
- 🔲 `motif` launches without error and shows welcome screen
- 🔲 `/help` lists all commands
- 🔲 Unknown slash command prints friendly error
- 🔲 `exit` saves empty history and exits cleanly
- 🔲 `rag/types.py` all dataclasses import without error
- 🔲 `rag/ingestion/__init__.py` exports `ingest_path`, `remove_document`, `sync_directory` (stubs)
- 🔲 `tests/conftest.py` fixtures run without error (`pytest --collect-only`)

---

## Phase 1 — Foundation

**Goal:** Working text QA via the REPL. Answerable questions return correct, grounded answers. ≥ 70% accuracy checkpoint.  
**Time-box:** 2 weeks (after Phase 0 complete)

### Tasks (Pipeline only — REPL/session done in Phase 0)

- 🔲 `rag/pipeline.py` — `QueryPipeline.answer()` end-to-end orchestration
- 🔲 `PyMuPDFParser` — text PDF extraction
- 🔲 `MarkdownParser` — heading-aware extraction
- 🔲 `SentenceChunker` — 512 token target, 64 token overlap
- 🔲 `Embedder` — nomic-embed ONNX INT8 wrapper, batch encode
- 🔲 `VectorStore` — Qdrant local mode, HNSW + sparse, `on_disk=True`
- 🔲 `BM25Index` — rank_bm25 wrapper, add/search/rebuild
- 🔲 `ChunkStore` — SQLite WAL, insert/fetch/delete_by_source
- 🔲 `IngestionTracker` — file hash tracking
- 🔲 RRF fusion (`fusion.py`)
- 🔲 `CrossEncoder` — MiniLM-L12 ONNX wrapper, rerank top-20 → top-5
- 🔲 `LLMClient` — llama-cpp-python wrapper, streaming
- 🔲 `ContextBuilder` — anti-middle ordering, history injection, token budget
- 🔲 `ModelManager` — lazy load/unload singleton
- 🔲 `prompts.py` — RAG_PROMPT, HISTORY_SYSTEM_PROMPT
- 🔲 Wire `/ingest`, `/status` slash commands into real pipeline
- 🔲 Unit tests: chunker, embedder, BM25, RRF, citation formatter
- 🔲 Unit tests: session history add/save/load/clear, rolling window trim
- 🔲 Integration test: ingest + ask end-to-end
- 🔲 Integration test: history persists across exit and relaunch

### Phase 1 Acceptance Checkpoint

- 🔲 `/ingest ./test_corpus/ -r` — no crash, progress shown
- 🔲 `/status` — correct counts
- 🔲 Plain-text query — streams answer with citations
- 🔲 Unanswerable question → refusal, no hallucination
- 🔲 `/ingest` again → 0 new chunks (deduplication)
- 🔲 History follow-up: "Expand on that" returns answer referencing prior turn
- 🔲 Exit + relaunch: welcome screen shows "Resuming previous session"
- 🔲 Manual accuracy check ≥ 70% on 20 test questions

---

## Phase 2 — Quality

**Goal:** Hit 85% RAGAS faithfulness on T2/T3.  
**Time-box:** 2 weeks (after Phase 1 checkpoint passes)

### Tasks

- 🔲 `SemanticChunker` — semantic-text-splitter binding, threshold 0.3
- 🔲 Enable semantic chunking on T2/T3 (`use_semantic = true`)
- 🔲 `QueryExpander` — HyDE prompt + routing heuristic (`should_use_hyde()`)
- 🔲 Enable adaptive HyDE on T2/T3
- 🔲 `cli.py remove PATH` command + `delete_document()`
- 🔲 `cli.py sync DIR` command + sync logic (add/delete/re-index)
- 🔲 Metadata filtering — `build_metadata_filter()` + Qdrant payload filter
- 🔲 CLI flags: `--file`, `--type`, `--pages`
- 🔲 Adjacent chunk merging in `ContextBuilder`
- 🔲 Extractive compression in `ContextBuilder`
- 🔲 bge-reranker-base ONNX for T3
- 🔲 Auto-calibrate relevance threshold on first run
- 🔲 Integration test: `sync` correctly adds and removes
- 🔲 Integration test: metadata filter restricts results
- 🔲 Run RAGAS on synthetic eval set

### Phase 2 Acceptance Checkpoint

- 🔲 RAGAS faithfulness ≥ 85% (T2/T3) on synthetic corpus eval
- 🔲 RAGAS faithfulness ≥ 75% (T1)
- 🔲 `cli.py remove` removes all chunks correctly (ING-19)
- 🔲 `cli.py sync` detects deleted files (ING-20)
- 🔲 Metadata filters working (RET-08 through RET-11)
- 🔲 Save as RAGAS baseline: `results/ragas_baseline.json`

---

## Phase 3 — Multimodal

**Goal:** All document types ingestible.  
**Time-box:** 2 weeks (after Phase 2 checkpoint passes)

### Tasks

- 🔲 `PaddleOCRParser` — image OCR for T2
- 🔲 `SuryaParser` — layout-aware OCR for T3
- 🔲 Update `PDFParser` to use OCR for scanned PDFs (T2/T3)
- 🔲 `DOCXParser` — python-docx, tables as markdown, headings as sections
- 🔲 `ImageParser` — PaddleOCR text extraction + image captioning gate
- 🔲 `AudioParser` — whisper.cpp via pywhispercpp, timestamps in metadata
- 🔲 `ModelManager.after_ingestion()` — unload ingestion models
- 🔲 Conditional moondream2 loading per image-heavy document
- 🔲 T3 opt-in: `cli.py setup --captioning`
- 🔲 Audio timestamp citation format verified
- 🔲 `is_academic_pdf()` heuristic for NOUGAT gate (T3 opt-in)
- 🔲 Integration test: audio chunk has start_time/end_time
- 🔲 Integration test: DOCX tables appear as markdown in chunks
- 🔲 Accuracy within 5% of text-only baseline on mixed corpus

---

## Phase 4 — Production Hardening

**Goal:** Latency targets met, stable across 1000 queries.  
**Time-box:** 2 weeks (after Phase 3 checkpoint passes)

### Tasks

- 🔲 ONNX model conversion for nomic-embed and reranker (verify INT8)
- 🔲 `Deduplicator` — SimHash near-duplicate detection at ingestion
- 🔲 Query result cache — SQLite, 500-query LRU
- 🔲 Cache privacy warning on startup
- 🔲 tantivy BM25 backend (auto-switch at 100K chunks)
- 🔲 `rag/evaluation/ragas_runner.py` — full RAGAS offline evaluation
- 🔲 `rag/evaluation/test_generator.py` — synthetic QA generation
- 🔲 `rag/evaluation/latency_test.py` — P50/P95 latency measurement
- 🔲 `rag/evaluation/ab_test.py` — A/B configuration comparison
- 🔲 `--consistency` flag (3× generation)
- 🔲 Full regression test suite passes
- 🔲 Logging to `~/.ragdb/motif.log`

### Phase 4 Acceptance Checkpoint

- 🔲 P95 latency ≤ 8s (T2), ≤ 5s (T3) over 100 queries
- 🔲 All TRD non-functional requirements verified (NFR-01 through NFR-10)
- 🔲 Stable across 1000 diverse queries (no crashes, no memory leaks)
- 🔲 RAGAS regression: all metrics within 2% of baseline

---

## Phase 5 — Optional Enhancements

**Status:** Not planned until Phase 4 complete.

- 🔲 RAPTOR hierarchical indexing (>500 pages)
- 🔲 Parent-document retrieval (optional, double index)
- 🔲 FLARE iterative retrieval (pending llama.cpp logit API)
- 🔲 NOUGAT academic PDF parser (T3 opt-in)
- 🔲 REST API wrapper
- 🔲 Desktop GUI (Tauri or Electron)

---

## Metrics Snapshots

*Updated after each phase checkpoint.*

| Date | Phase | Faithfulness | Relevancy | P95 Latency | Disk | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | No checkpoint yet |

---

## Active Blockers

*None currently.*

| # | Description | Owner | Opened | Status |
|---|---|---|---|---|
| — | — | — | — | — |

---

## Deferred Decisions Log

*Decisions punted during implementation. Revisit at the noted phase.*

| Decision | Punted By | Revisit At | Context |
|---|---|---|---|
| HyDE vs multi-query: which is better on real corpus | Pre-impl analysis | Phase 2 checkpoint | A/B test on synthetic QA after Phase 2 |
| Optimal relevance threshold for corpus | Pre-impl analysis | After first ingest | Auto-calibration uses 0.3 default until calibrated |
| Switch to tantivy for BM25 | Phase 1 implementation | Phase 4 (or when >100K chunks) | rank_bm25 sufficient for MVP corpus |
| Parent-document retrieval enable/disable | Phase 2 | Phase 3 (if recall issues reported) | Storage cost 2×; current stack may be sufficient |
| bge-reranker-base for T2 | Review discussion | Phase 2 (evaluate after RAGAS) | Small gain, 150 MB extra — validate first |
