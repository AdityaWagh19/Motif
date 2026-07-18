# MVP Scope — Motif Offline Multimodal RAG

> **Depends on:** `context.md`, `trd.md`  
> **Time-box:** 2 weeks (Phases 1–2 from the delivery plan)  
> **Goal:** A working, query-able text RAG system that can be evaluated against the 70% accuracy checkpoint.

---

## 1. MVP Is In Scope

These are the only features to implement in the MVP. Anything not listed here is out of scope for Phase 1–2.

### Core Pipeline

- [x] `cli.py ask QUERY` — returns a streamed answer with citations
- [x] `cli.py ingest PATH` — ingests PDF (text-only, no OCR) and Markdown files
- [x] `cli.py status` — shows document count, chunk count, storage size

### Ingestion

- [x] **PyMuPDF** text extraction (no OCR, no scanned PDF support)
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

### CLI

- [x] Basic Rich progress during ingestion
- [x] Citations displayed after answer
- [x] `--no-hyde` flag (HyDE is off by default in MVP; can be toggled)
- [x] Error messages with actionable guidance

### Config

- [x] `config.toml` loaded at startup
- [x] Auto-tier detection (T1/T2/T3)
- [x] Manual tier override in config

---

## 2. MVP Is NOT In Scope

These features are explicitly deferred. Do not implement them in Phase 1–2.

| Feature | Deferred To |
|---|---|
| OCR (PaddleOCR, Surya) | Phase 3 |
| DOCX parser | Phase 3 |
| Audio ingestion (whisper.cpp) | Phase 3 |
| Image ingestion | Phase 3 |
| moondream2 captioning | Phase 3 |
| Semantic chunking | Phase 2 (end of Phase 2) |
| HyDE query expansion | Phase 2 |
| `cli.py remove` | Phase 2 |
| `cli.py sync` | Phase 2 |
| Metadata filtering (`--file`, `--type`, `--pages`) | Phase 2 |
| Adjacent chunk merging | Phase 2 |
| Extractive compression | Phase 2 |
| Query cache | Phase 4 |
| Deduplicator | Phase 4 |
| RAGAS evaluation runner | Phase 4 |
| Synthetic QA generation | Phase 4 |
| tantivy (large corpus BM25) | Phase 4 |
| RAPTOR hierarchical indexing | Phase 5 |
| Parent-document retrieval | Phase 3 (optional) |
| NOUGAT parser | Phase 3 (optional, T3) |
| `--consistency` flag (3× generation) | Phase 4 |
| bge-reranker-base (T3 upgrade) | Phase 2 |
| Desktop GUI | Post-Phase 4 |
| REST API | Post-Phase 4 |

---

## 3. MVP Acceptance Tests

The MVP is complete when **all of the following commands work** on at least two real PDF documents and one Markdown file:

```bash
# 1. Ingest documents
python cli.py ingest ./test_corpus/ --recursive
# Expected: Progress bar, "Ingested N files", no crash

# 2. Check status
python cli.py status
# Expected: Documents: N, Chunks: M, Storage: X MB, Tier: T1/T2/T3

# 3. Ask a question answerable from the corpus
python cli.py ask "What is the main finding of the paper?"
# Expected: Streamed answer with at least 1 citation from the ingested documents
# Answer must NOT reference information not in the documents

# 4. Ask an unanswerable question
python cli.py ask "What is the capital of France?"
# Expected: "I could not find relevant information in the documents" or similar refusal
# Must NOT hallucinate a confident answer

# 5. Re-ingest (test deduplication)
python cli.py ingest ./test_corpus/
# Expected: "0 new files indexed" or equivalent — chunks not duplicated

# 6. Test --no-hyde flag
python cli.py ask "Define the key terminology" --no-hyde
# Expected: Works correctly (HyDE not used)
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
