# MVP Scope — Motif Offline Multimodal RAG

> **Depends on:** `context.md`, `trd.md`  
> **Phases:** Phase 0 (Infrastructure) + Phase 1 (Text RAG) + Phase 2 (Quality)  
> **Goal:** A working, query-able text RAG system accessible via REPL. Phase 0 delivers the installable shell; Phase 1 delivers the working pipeline inside it.

---

## 0. Phase 0 — Infrastructure (prerequisite, not time-boxed separately)

These files are created before any pipeline code. They must exist before Phase 1 begins.

- [ ] `pyproject.toml` — package definition, `motif` entry point, pinned deps
- [ ] `install.sh` — Linux/macOS bootstrap (uv + CUDA detection)
- [ ] `install.ps1` — Windows PowerShell bootstrap
- [ ] `config.template.toml` — fully commented config with T1/T2/T3 variants shown
- [ ] `rag/config.py` — config dataclasses, `detect_hardware_tier()`
- [ ] `rag/types.py` — **all shared dataclasses** (`Chunk`, `ScoredPassage`, `Citation`, `AnswerResult`, `IngestResult`, `SyncResult`) — **must exist before any Phase 1 code**
- [ ] `rag/models/__init__.py` — package init
- [ ] `rag/models/embedder.py` — `Embedder` class skeleton (ONNX wrapper, no Phase 1 logic yet)
- [ ] `rag/models/reranker.py` — `Reranker` class skeleton
- [ ] `rag/session.py` — Session class: history list, `add_turn()`, `save()`, `load()`, `clear()`, `new()`, rolling window trim
- [ ] `cli.py` — prompt_toolkit REPL: welcome screen (Rich panel), REPL loop, slash command router, session load/save
- [ ] `rag/commands/__init__.py` — command registry
- [ ] `rag/commands/help.py` — `/help`
- [ ] `rag/commands/clear.py` — `/clear`, `/new`
- [ ] `rag/commands/status.py` — `/status` stub
- [ ] `rag/commands/ingest.py` — `/ingest` stub
- [ ] `rag/commands/remove.py` — `/remove` stub
- [ ] `rag/commands/sync.py` — `/sync` stub
- [ ] `rag/commands/setup.py` — `/setup` stub
- [ ] `rag/ingestion/__init__.py` — **public API stubs**: `ingest_path()`, `remove_document()`, `sync_directory()` with type signatures from `rag.types` — not implemented yet, raises `NotImplementedError`
- [ ] `setup_models.py` — model download with tier detection and progress bars
- [ ] `tests/conftest.py` — shared fixtures: `tmp_db_root`, `sample_pdf`, `sample_md`, `minimal_config`
- [ ] `models/.gitkeep`, `tests/unit/.gitkeep`, `tests/integration/.gitkeep`

---

## 1. MVP Is In Scope (Phase 1 pipeline)

### Ingestion

- [x] **PyMuPDF** text extraction (no OCR, no scanned PDF support in Phase 1)
- [x] **Markdown** parser (markdown-it-py)
- [x] **SentenceChunker** — sentence-boundary split, 512 tokens target, 64-token overlap (no semantic chunking)
- [x] **nomic-embed-text-v1.5 ONNX INT8** — batch embedding at ingestion
- [x] **Qdrant local mode** — HNSW dense index + sparse vectors (`on_disk=True`)
- [x] **rank_bm25** — BM25 lexical index (in-memory)
- [x] **SQLite chunk store** — WAL mode, ChunkMetadata stored
- [x] **Ingestion tracker** — file hash tracking, skip already-indexed files

### Retrieval

- [x] **Hybrid retrieval** — dense (Qdrant) + sparse (Qdrant) + BM25, fused with RRF (k=60)
- [x] **Top-k retrieval:** T1=20, T2=25, T3=30 candidates

### Reranking

- [x] **MiniLM-L6 ONNX** (T1) / **MiniLM-L12 ONNX** (T2/T3)
- [x] **Relevance threshold** — default 0.3, drops passages below threshold
- [x] **Top-k rerank:** T1=3, T2/T3=5 final passages

### Generation

- [x] **Phi-3.5-mini Q4_K_M** (T1) or **Qwen2.5-7B Q4_K_M** (T2/T3) via llama-cpp-python
- [x] **Anti-middle ordering** of context passages
- [x] **Streaming output** — rich Live display
- [x] **System prompt** — explicit "answer only from context" instruction
- [x] **Temperature = 0.1**

### Core Pipeline (Phase 1)

- [ ] PDF (text-only) + Markdown ingestion via `/ingest`
- [ ] Streamed answer with citations via plain-text query at the REPL prompt
- [ ] `/status` slash command
- [ ] Conversation history: rolling 3-turn window, persisted to `~/.ragdb/history.json`

---

## 2. MVP Is NOT In Scope

These features are explicitly deferred. Do not implement them in Phase 1–2.

| Feature | Status / Phase |
|---|---|
| OCR (PaddleOCR) | ✅ Done (Phase 3) |
| DOCX parser | ✅ Done (Phase 3) |
| Audio ingestion (whisper.cpp) | ✅ Done (Phase 3) |
| Image ingestion | ✅ Done (Phase 3) |
| Semantic chunking | ✅ Done (Phase 4) |
| HyDE query expansion | ✅ Done (Phase 4) |
| `motif remove` command | ✅ Done (Phase 2) |
| `motif sync` command | ✅ Done (Phase 2) |
| Metadata filtering (`/file`, `/type`, `/pages`) | ✅ Done (Phase 2) |
| Adjacent chunk merging | ✅ Done (Phase 2) |
| Extractive compression | ✅ Done (Phase 2) |
| Query cache | ✅ Done (Phase 4) |
| Deduplicator | ✅ Done (Phase 4) |
| RAGAS evaluation runner | ✅ Done (Phase 6) |
| Synthetic QA generation | ✅ Done (Phase 6) |
| Intent Classification | ✅ Done (Phase 7) |
| bge-reranker-base (T3 upgrade) | ✅ Done (Phase 4) |
| RAPTOR hierarchical indexing | Deferred to Phase 8 |
| Parent-document retrieval | Deferred |
| NOUGAT/Surya parsers | Dropped (PaddleOCR is sufficient) |
| Desktop GUI / REST API | Deferred |

---

## 2. MVP Acceptance Tests

The MVP is complete when **all of the following work** on at least two real PDF documents and one Markdown file.

```bash
# 1. Install (Phase 0 gate)
motif
# Expected: welcome screen renders with tier, model, chunk count

# 2. Ingest documents (Phase 1 gate)
# At the motif > prompt:
/ingest ./test_corpus/ -r
# Expected: progress bar, "Ingested N files"

# 3. Check status
/status
# Expected: Documents: N, Chunks: M, Storage: X MB, Tier: T1/T2/T3

# 4. Ask an answerable question (plain text at prompt)
What is the main finding of the paper?
# Expected: streamed answer with at least 1 citation

# 5. Ask an unanswerable question
What is the capital of France?
# Expected: refusal ("not found in documents"), no confident hallucination

# 6. Test conversation history
What methodology was used?
# [answer]
Expand on the sampling approach.
# Expected: answer references the prior context (the previous Q&A is in history)

# 7. Exit and restart
exit
motif
# Expected: welcome screen shows "Resuming previous session" with last query shown

# 8. Start fresh
/new
# Expected: history cleared, session starts clean

# 9. Re-ingest (deduplication)
/ingest ./test_corpus/
# Expected: 0 new chunks added
```

---

## 4. MVP Configuration (config.toml snippet)

```toml
# Minimal MVP config — copy to project root

[hardware]
tier = "auto"   # auto-detect; override: "T1", "T2", "T3"

[models]
llm_path    = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"  # or Phi-3.5-mini
embed_model = "models/nomic-embed-text-v1.5"
reranker    = "models/ms-marco-MiniLM-L-12-v2"

[llm]
n_gpu_layers = 20      # T2 default; set 0 for T1, 28 for T3
ctx_size     = 3072
max_tokens   = 400
temperature  = 0.1
threads      = 6

[retrieval]
top_k_retrieval = 25
top_k_rerank    = 5
relevance_threshold = 0.3
query_expansion = "none"   # MVP: HyDE off; "hyde" in Phase 2

[chunking]
target_tokens  = 512
overlap_tokens = 64
use_semantic   = false     # Phase 2 feature

[generation]
context_max_tokens = 2048
streaming          = true

[storage]
db_path              = "~/.ragdb"
query_cache_enabled  = false   # Phase 4 feature
```

---

## 5. MVP Accuracy Checkpoint

Before moving to Phase 2, verify:

> **Target: ≥ 70% accuracy** on a manually assembled 20-question test set from your ingested corpus.

Measure informally — read the answer, check if it is correct and supported by the cited passage.

If accuracy is below 60%:
- Check that the cited passage actually contains the answer (retrieval failure vs generation failure)
- Enable `--verbose` mode to inspect retrieved passages
- Tune `relevance_threshold` downward (try 0.2)
- Verify the LLM model file is Q4_K_M and not a different quantization

If accuracy is above 70%: **proceed to Phase 2.**

---

## 6. Phase Delivery Plan (Reference)

| Phase | Goal | Weeks | Key Additions |
|---|---|---|---|
| **1 — Foundation** | Text RAG via CLI | 1 | Core pipeline: PDF + MD + sentence chunk + embed + Qdrant + BM25 + MiniLM + LLM |
| **2 — Quality** | Hit 85% faithfulness | 2 | Semantic chunking, HyDE, `remove`, `sync`, metadata filters, adjacent merge, extractive compress, bge-reranker (T3) |
| **3 — Multimodal** | All modality ingestion | 2 | PaddleOCR, Surya, whisper.cpp, DOCX, image parser, moondream2 opt-in |
| **4 — Production** | Latency + reliability | 2 | ONNX conversion, dedup, query cache, RAGAS eval, tantivy, logging, `--consistency` |
| **5 — Optional** | Large corpora + advanced | TBD | RAPTOR, parent-doc retrieval, FLARE, GUI, REST API |
   
 