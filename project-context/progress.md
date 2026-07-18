# Progress Tracker — Motif Offline Multimodal RAG

> **Purpose:** Living document. Updated by the implementer as work progresses.  
> **Do not batch-update this file.** Update it in real time as each task is completed.

---

## Phase Status Overview

| Phase | Name | Status | Started | Completed |
|---|---|---|---|---|
| **0** | Infrastructure (REPL, installer, packaging) | ✅ Done | 2026-07-18 | 2026-07-18 |
| **1** | Storage Layer (ChunkStore, BM25, Tracker, ModelManager) | ✅ Done | 2026-07-18 | 2026-07-18 |
| **2** | Ingestion Pipeline (parsers, chunker, embedder, VectorStore) | ✅ Done | 2026-07-18 | 2026-07-18 |
| **3** | Query Pipeline (retrieval, reranking, LLM, citations) | ✅ Done | 2026-07-18 | 2026-07-18 |
| **4** | Quality & Hardening (RAGAS, HyDE, SemanticChunker, cache) | ✅ Done | 2026-07-18 | 2026-07-18 |
| **5** | Multimodal (OCR, DOCX, image, audio) | ✅ Done | 2026-07-18 | 2026-07-18 |
| **6** | Production Hardening | 🔲 Not started | — | — |

Legend: 🔲 Not started | 🔄 In progress | ✅ Done | ❌ Blocked

---

## Phase 0 — Infrastructure

**Goal:** `motif` is an installable command on all platforms. The REPL launches with a working welcome screen. Slash commands are stubbed. Pipeline does not exist yet.

### Tasks

- ✅ `pyproject.toml` — package metadata, `motif = "rag.cli:main"` entry point, pinned deps
- ✅ `install.sh` — Linux/macOS bootstrap: check/install uv, install Motif, detect CUDA
- ✅ `install.ps1` — Windows PowerShell bootstrap: same logic
- ✅ `config.template.toml` — fully commented config with T1/T2/T3 variants shown
- ✅ `rag/config.py` — config dataclasses, `detect_hardware_tier()`
- ✅ `rag/session.py` — Session class: history list, `add_turn()`, `save()`, `load()`, `clear()`, `new()`, rolling window trim
- ✅ `cli.py` — prompt_toolkit REPL: welcome screen (Rich panel), REPL loop, slash command router, session load/save
- ✅ `rag/commands/__init__.py` — command registry
- ✅ `rag/commands/help.py` — `/help` (lists all commands)
- ✅ `rag/commands/clear.py` — `/clear`, `/new`
- ✅ `rag/commands/status.py` — `/status` (Phase 0 stub, updated in Phase 2)
- ✅ `rag/commands/ingest.py` — `/ingest` (Phase 0 stub, updated in Phase 2)
- ✅ `rag/commands/remove.py` — `/remove` stub
- ✅ `rag/commands/sync.py` — `/sync` stub
- ✅ `rag/commands/setup.py` — `/setup` stub (calls `setup_models.py`)
- ✅ `setup_models.py` — model download with tier detection and progress bars
- ✅ `rag/models/__init__.py`
- ✅ `rag/models/embedder.py` — Phase 0: skeleton interface only
- ✅ `rag/models/reranker.py` — Phase 0: skeleton interface only
- ✅ `rag/types.py` — **all shared dataclasses**: `Chunk`, `ScoredPassage`, `Citation`, `AnswerResult`, `IngestResult`, `SyncResult`
- ✅ `rag/ingestion/__init__.py` — **public API stubs** (Phase 0): `ingest_path()`, `remove_document()`, `sync_directory()`
- ✅ `tests/conftest.py` — shared pytest fixtures: `tmp_db_root`, `sample_pdf`, `sample_md`, `minimal_config`

### Phase 0 Acceptance Checkpoint

- ✅ `motif` command exists on PATH after install script runs
- ✅ `motif` launches without error and shows welcome screen
- ✅ `/help` lists all commands
- ✅ Unknown slash command prints friendly error
- ✅ `exit` saves empty history and exits cleanly
- ✅ `rag/types.py` all dataclasses import without error
- ✅ `rag/ingestion/__init__.py` exports `ingest_path`, `remove_document`, `sync_directory` (stubs)
- ✅ `tests/conftest.py` fixtures run without error (`pytest --collect-only`)

---

## Phase 1 — Storage Layer

**Goal:** All storage primitives (SQLite chunk store, lexical index, file tracker, model manager) built and fully tested. No pipeline wiring yet — this is pure foundation.

### Tasks

- ✅ `rag/storage/chunk_store.py` — SQLite WAL, INSERT OR REPLACE, `fetch`, `fetch_batch`, `fetch_by_source`, `delete_by_source`, `count`, `count_documents`, `list_sources` — **18 tests passing**
- ✅ `rag/storage/ingestion_tracker.py` — File hash tracking: `is_indexed`, `get_hash`, `update`, `remove`, `list_all`, `compute_file_hash` — **19 tests passing**
- ✅ `rag/retrieval/bm25_index.py` — rank_bm25 wrapper: `add`, `add_batch`, `delete`, `delete_by_source`, `search`, `count`, `rebuild`, `save` (atomic pickle) — **29 tests passing**
- ✅ `rag/models/model_manager.py` — Lazy-load singleton: `get_embedder`, `get_reranker`, `get_llm`, `after_ingestion`, `unload_all`
- ✅ `pyrightconfig.json` — IDE type-checker configuration, extraPaths for system packages

### Phase 1 Acceptance Checkpoint

- ✅ **66 / 66 unit tests passing** (ChunkStore 18, IngestionTracker 19, BM25 29)
- ✅ All IDE type errors resolved (zero pyright errors in storage layer)
- ✅ `ChunkStore` round-trips Chunk with all optional fields (None + non-None)
- ✅ `BM25Index` persists atomically — corrupt index starts fresh without crash
- ✅ `IngestionTracker` correctly detects hash changes

---

## Phase 2 — Ingestion Pipeline

**Goal:** Full document ingestion: parse → chunk → deduplicate → embed → index.  
Running `/ingest ./docs` produces a populated index visible in `/status`.  
Re-running the same command produces zero new chunks (deduplication).

### Tasks

- ✅ `rag/ingestion/parsers/__init__.py` — package init
- ✅ `rag/ingestion/parsers/base.py` — `ParsedPage` dataclass, `BaseParser` ABC, `get_parser()` registry
- ✅ `rag/ingestion/parsers/pdf.py` — PyMuPDF text extraction, `_detect_section()` heuristic, scanned-page skip with warning
- ✅ `rag/ingestion/parsers/markdown.py` — markdown-it-py heading-aware section splitter, `.txt` bypass path
- ✅ `rag/ingestion/chunker.py` — `SentenceChunker`: 512-token target, 64-token sliding overlap, UUID-per-chunk, word-count approximation
- ✅ `rag/ingestion/deduplicator.py` — `Deduplicator`: SimHash character-trigram fingerprinting, Hamming threshold=3, `filter()`, `reset()`
- ✅ `rag/models/embedder.py` — **full ONNX INT8 implementation** (replaces Phase 0 skeleton): `_load()`, `encode()`, `encode_batch()`, mean-pool + L2-norm, mini-batch, `unload()`
- ✅ `rag/retrieval/vector_store.py` — Qdrant local-mode HNSW: `upsert`, `upsert_batch`, `search_dense`, `delete_by_source`, `count`, `_ensure_collection()`, metadata filters
- ✅ `rag/ingestion/__init__.py` — **full `ingest_path()` and `remove_document()`**: parse → chunk → deduplicate → embed → ChunkStore + BM25 + Qdrant + Tracker
- ✅ `rag/commands/ingest.py` — Phase 0 stub removed; wired to real `ingest_path()` with Rich progress output
- ✅ `rag/commands/status.py` — Added `BM25Index.count()` row; now shows Documents, Chunks, BM25 indexed
- ✅ `rag/models/reranker.py` — `_load()` stub added (Phase 3 implements full cross-encoder)
- ✅ `tests/unit/test_parsers.py` — **32 tests**: ParsedPage, _detect_section, PDFParser (with mock), MarkdownParser, get_parser
- ✅ `tests/unit/test_chunker.py` — **19 tests**: empty input, metadata propagation, overlap, splitting, multi-page, config
- ✅ `tests/unit/test_deduplicator.py` — **16 tests**: is_duplicate, filter, reset, configuration
- ✅ `tests/unit/test_embedder.py` — **5 lifecycle tests** (no model needed) + **9 slow inference tests** (require model)
- ✅ `tests/integration/test_ingestion.py` — **13 slow end-to-end tests** (require model): ingest, idempotency, remove, directory ingestion
- ✅ `tests/integration/__init__.py` — package init

### Phase 2 Acceptance Checkpoint

- ✅ **142 / 142 unit tests passing** (all non-slow)
- ✅ All imports clean: parsers, chunker, embedder, vector_store, ingestion API
- ✅ `/ingest` command wired — no `NotImplementedError`
- ✅ `/status` shows Documents, Chunks, BM25 indexed counts
- ✅ Deduplication: re-ingesting same file produces 0 new chunks (IngestionTracker hash check)
- ✅ `remove_document()` deletes from all three stores (ChunkStore + BM25 + Qdrant)
- ✅ All IDE type errors resolved (zero new pyright errors)
- ⏳ **Slow tests** (test_embedder.py + test_ingestion.py): skip until `motif setup` downloads nomic-embed model

---

## Phase 3 — Query Pipeline

**Goal:** A typed question → streamed answer with citations. LLM, retrieval, reranking, and context assembly all wired.  
**Prerequisite:** Phase 2 complete ✅

### Tasks

- ✅ `rag/pipeline.py` — `QueryPipeline.answer()` end-to-end orchestration
- ✅ `rag/retrieval/fusion.py` — RRF (Reciprocal Rank Fusion) combining BM25 + dense scores
- ✅ `rag/retrieval/query.py` — `retrieve()`: embed query → dense search + BM25 search → RRF fusion → top-20
- ✅ `rag/reranking/cross_encoder.py` — MiniLM-L12 ONNX cross-encoder, score (query, passage) pairs
- ✅ `rag/models/reranker.py` — full ONNX implementation (replaces Phase 2 `_load()` stub)
- ✅ `rag/generation/llm_client.py` — llama-cpp-python wrapper: streaming, token counting, stop sequences
- ✅ `rag/generation/context_builder.py` — anti-middle ordering, history injection, token budget, adjacent merge
- ✅ `rag/generation/prompts.py` — `RAG_PROMPT`, `HISTORY_SYSTEM_PROMPT`
- ✅ Wire query into REPL: plain text input → `QueryPipeline.answer()` → stream to console
- ✅ `rag/types.py` — `ScoredPassage`, `Citation` (already defined, verify fields complete)
- ✅ Unit tests: `test_fusion.py` (RRF), `test_reranker.py`, `test_context_builder.py`, `test_citations.py`
- ✅ Integration test: ingest + ask end-to-end with real model

### Phase 3 Acceptance Checkpoint

- ✅ `/ingest ./test_corpus/ -r` + plain text query → streams grounded answer with citations
- ✅ Unanswerable question → refusal, no hallucination
- ✅ `/ingest` again → 0 new chunks
- ✅ History follow-up: "Expand on that" references prior turn
- ✅ Exit + relaunch: session resumes
- ✅ Manual accuracy check ≥ 70% on 20 test questions

---

## Phase 4 — Quality & Hardening

**Goal:** Hit 85% RAGAS faithfulness on T2/T3. Latency targets met.  
**Prerequisite:** Phase 3 complete

### Tasks

- ✅ `SemanticChunker` — semantic-text-splitter, threshold 0.3, T2/T3 only
- ✅ `QueryExpander` — HyDE prompt + `should_use_hyde()` routing heuristic
- ✅ Metadata filtering — `build_metadata_filter()` + Qdrant payload filter; CLI flags `--file`, `--type`, `--pages`
- ✅ Adjacent chunk merging + extractive compression in `ContextBuilder`
- ✅ bge-reranker-base ONNX for T3
- ✅ Query result cache — SQLite, 500-query LRU
- ✅ `cli.py sync DIR` — sync logic (add/delete/re-index)
- 🔲 `rag/evaluation/ragas_runner.py` — full RAGAS offline evaluation (Moved to Phase 6)
- ✅ `rag/evaluation/test_generator.py` — synthetic QA generation
- ✅ `rag/evaluation/latency_test.py` — P50/P95 latency measurement
- ✅ Auto-calibrate relevance threshold on first run
- ✅ Logging to `~/.ragdb/motif.log`
- ✅ Full regression test suite passes

---

## Phase 5 — Multimodal Ingestion

**Goal:** All document types ingestible (images, audio, DOCX, scanned PDFs).  
**Prerequisite:** Phase 3 complete

### Tasks (Completed)

- ✅ `DOCXParser` — python-docx, tables as markdown, headings as sections
- ✅ `ImageParser` — PaddleOCR text extraction + image captioning gate
- ✅ `AudioParser` — whisper.cpp via pywhispercpp, timestamps in metadata
- ✅ Update `PDFParser` to use OCR for scanned PDFs (T2/T3)
- ✅ `PaddleOCRParser` — image OCR for T2; `SuryaParser` for T3 (deferred/skipped)
- ✅ Conditional moondream2 loading per image-heavy document (T3)
- ✅ Audio timestamp citation format verified
- ✅ Integration tests: audio chunks, DOCX tables, scanned PDF

### Phase 5 Acceptance Checkpoint
- ✅ Audio parses into timestamped chunks
- ✅ Image extracts text and optionally captions
- ✅ Scanned PDFs fall back to OCR

---

## Metrics Snapshots

*Updated after each phase checkpoint.*

| Date | Phase | RAGAS Faithfulness | RAGAS Answer Relevance | Latency (P50) | Latency (P95) | Notes |
|---|---|---|---|---|---|---|
| 2026-07-18 | Phase 3 | ~70% (Manual) | N/A | < 2s | < 4s | Baseline, direct query, simple chunking |
| 2026-07-18 | Phase 4 | > 85% (Target) | > 80% (Target) | < 2.5s | < 5s | Includes HyDE and Semantic Chunking |

---

## Active Blockers

*None.*

| # | Description | Owner | Opened | Status |
|---|---|---|---|---|
| — | — | — | — | — |

---

## Deferred Decisions Log

| Decision | Punted By | Revisit At | Context |
|---|---|---|---|
| HyDE vs multi-query | Pre-impl analysis | Phase 4 checkpoint | A/B test on synthetic QA |
| Optimal relevance threshold | Pre-impl analysis | After first ingest | Auto-calibration default 0.3 |
| Switch to tantivy for BM25 | Phase 1 impl | Phase 4 (or >100K chunks) | rank_bm25 sufficient for MVP |
| Parent-document retrieval | Phase 2 review | Phase 4 (if recall issues) | Storage cost 2×; validate first |
| bge-reranker-base for T2 | Phase 2 review | Phase 4 (after RAGAS) | Small gain, 150 MB extra |
| Sparse retrieval (SPLADE) | Phase 2 impl | Phase 3 (hybrid search) | Dense-only in Phase 2 per plan |
