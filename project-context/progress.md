# Progress Tracker — Motif Offline Multimodal RAG

> **Purpose:** Living document. Updated by the implementer as work progresses.  
> **Do not batch-update this file.** Update it in real time as each task is completed.

---

## Phase Status Overview

| Phase | Name | Status | Started | Completed |
|---|---|---|---|---|
| **1** | Foundation (Text RAG MVP) | 🔲 Not started | — | — |
| **2** | Quality (85% faithfulness) | 🔲 Not started | — | — |
| **3** | Multimodal Ingestion | 🔲 Not started | — | — |
| **4** | Production Hardening | 🔲 Not started | — | — |
| **5** | Optional Enhancements | 🔲 Not started | — | — |

Legend: 🔲 Not started | 🔄 In progress | ✅ Done | ❌ Blocked

---

## Phase 1 — Foundation

**Goal:** Working single-document text QA via CLI. ≥ 70% accuracy checkpoint.  
**Time-box:** 2 weeks

### Tasks

- 🔲 Project scaffolding: directory structure, `__init__.py`, `config.py`
- 🔲 `config.toml` template + `detect_hardware_tier()`
- 🔲 `PyMuPDFParser` — text PDF extraction
- 🔲 `MarkdownParser` — heading-aware markdown extraction
- 🔲 `SentenceChunker` — 512 token target, 64 token overlap
- 🔲 `Embedder` — nomic-embed ONNX INT8 wrapper, batch encode
- 🔲 `VectorStore` — Qdrant local mode, HNSW + sparse, `on_disk=True`
- 🔲 `BM25Index` — rank_bm25 wrapper, add/search/rebuild
- 🔲 `ChunkStore` — SQLite WAL, insert/fetch/delete_by_source
- 🔲 `IngestionTracker` — file hash tracking
- 🔲 RRF fusion (`fusion.py`)
- 🔲 `CrossEncoder` — MiniLM-L12 ONNX wrapper, rerank top-20 → top-5
- 🔲 `LLMClient` — llama-cpp-python wrapper, streaming
- 🔲 `ContextBuilder` — anti-middle ordering, token budget check
- 🔲 `prompts.py` — RAG_PROMPT, system prompt
- 🔲 `QueryPipeline.answer()` — end-to-end orchestration
- 🔲 `cli.py` — `ingest`, `ask`, `status` commands
- 🔲 `ModelManager` — lazy load/unload skeleton
- 🔲 Unit tests: chunker, embedder, BM25, RRF, citation formatter
- 🔲 Integration test: ingest + ask end-to-end

### Phase 1 Acceptance Checkpoint

- 🔲 `cli.py ingest ./test_corpus/` — no crash, progress shown
- 🔲 `cli.py status` — correct counts
- 🔲 `cli.py ask "..."` — streams answer with citations
- 🔲 Unanswerable question → refusal (no hallucination)
- 🔲 Re-ingest → 0 new chunks
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
