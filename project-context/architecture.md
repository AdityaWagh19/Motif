# Architecture — Motif Offline Multimodal RAG

> **Depends on:** `context.md`  
> **Consumed by:** `flow.md`, `trd.md`, `instructions.md`

---

## 1. System Overview

Motif is a single-user, offline CLI application. There is no server process, no network calls, and no shared state between users. All components run in a single Python process (plus llama.cpp as a subprocess or via python bindings).

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                            │
│           click + rich  |  streaming output  |  citations       │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
    ┌──────────▼──────────┐        ┌──────────▼──────────┐
    │   Ingestion Pipeline │        │   Query Pipeline     │
    │  (one-time, offline) │        │  (per-query, ~2-11s) │
    └──────────┬──────────┘        └──────────┬──────────┘
               │                              │
    ┌──────────▼──────────────────────────────▼──────────┐
    │                   Storage Layer                      │
    │   Qdrant (HNSW + sparse)  |  rank_bm25  |  SQLite   │
    └─────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │    Model Layer       │
    │  LLM | Embed | Rank  │
    │  OCR | Audio | VLM   │
    └─────────────────────┘
```

---

## 2. Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|---|---|---|
| `cli.py` | Entry point; routes commands to pipeline | `ingest()`, `ask()`, `sync()`, `status()`, `remove()` |
| `rag/config.py` | Config dataclasses; TOML loading; tier detection | `RAGConfig`, `detect_hardware_tier()` |
| `rag/pipeline.py` | End-to-end query orchestration | `QueryPipeline.answer()` |
| `rag/ingestion/parsers/` | Modality-specific document parsing | `PDFParser`, `DOCXParser`, `MarkdownParser`, `ImageParser`, `AudioParser` |
| `rag/ingestion/chunker.py` | Semantic / sentence chunking | `SemanticChunker`, `SentenceChunker` |
| `rag/ingestion/embedder.py` | nomic-embed ONNX inference | `Embedder.encode()`, `Embedder.encode_batch()` |
| `rag/ingestion/deduplicator.py` | Near-duplicate chunk detection | `Deduplicator.is_duplicate()` |
| `rag/retrieval/vector_store.py` | Qdrant HNSW + sparse search wrapper | `VectorStore.search_dense()`, `.search_sparse()`, `.delete()` |
| `rag/retrieval/bm25_index.py` | BM25 lexical search | `BM25Index.search()`, `.add()`, `.delete()`, `.rebuild()` |
| `rag/retrieval/fusion.py` | Reciprocal Rank Fusion | `rrf_fuse(dense, sparse, bm25, k=60)` |
| `rag/retrieval/expander.py` | HyDE and multi-query expansion | `QueryExpander.expand()` |
| `rag/reranking/cross_encoder.py` | MiniLM / bge-reranker ONNX scoring | `CrossEncoder.rerank(query, passages, top_k)` |
| `rag/generation/llm_client.py` | llama.cpp inference wrapper | `LLMClient.generate()`, `.stream()` |
| `rag/generation/context_builder.py` | Context assembly, ordering, merging | `ContextBuilder.build()` |
| `rag/generation/prompts.py` | Prompt templates for all query types | `RAG_PROMPT`, `HYDE_PROMPT`, `SYNTHESIS_PROMPT` |
| `rag/storage/chunk_store.py` | SQLite CRUD for chunk text + metadata | `ChunkStore.insert()`, `.fetch()`, `.delete_by_source()` |
| `rag/storage/ingestion_tracker.py` | File hash tracking for incremental ingestion | `IngestionTracker.is_indexed()`, `.update()`, `.remove()` |
| `rag/models/model_manager.py` | Lazy load / unload of all models | `ModelManager.get()`, `.load()`, `.unload()`, `.after_ingestion()` |
| `rag/evaluation/ragas_runner.py` | Offline RAGAS evaluation with local LLM | `run_evaluation(dataset, metrics)` |
| `rag/evaluation/test_generator.py` | Synthetic QA pair generation | `create_eval_dataset(chunks, llm, n=100)` |

---

## 3. Technology Stack

| Component | Library | Version (pinned) | Justification |
|---|---|---|---|
| **LLM inference** | llama-cpp-python | 0.3.x | CPU+GPU, mature offline inference, streaming |
| **Embedding** | ONNX Runtime + nomic-embed-text-v1.5 | ort 1.20.x | No torch dependency at query time; INT8 quantization |
| **Reranker** | ONNX Runtime + MiniLM-L12 / bge-reranker-base | ort 1.20.x | Same ONNX runtime, no extra dependency |
| **Vector store** | qdrant-client (local mode) | 1.12.x | HNSW + sparse in one library, no server process |
| **BM25 (small corpus)** | rank-bm25 | 0.2.x | Pure Python, zero setup |
| **BM25 (large corpus)** | tantivy-py | 0.22.x | Rust-backed, memory-mapped, >100K chunks |
| **PDF (text)** | pymupdf | 1.24.x | Fast, accurate text + layout extraction |
| **PDF (scanned, T2)** | paddleocr | 2.8.x | Better than Tesseract; runs on CPU |
| **PDF (scanned, T3)** | surya-ocr | 0.6.x | Layout-aware, best quality for complex PDFs |
| **DOCX** | python-docx | 1.1.x | Standard; handles tables, headers, footnotes |
| **Markdown** | markdown-it-py | 3.0.x | Accurate AST parse; handles GFM extensions |
| **Audio** | pywhispercpp (whisper.cpp bindings) | 1.2.x | CPU-efficient, quantized, offline |
| **Image captioning (T3)** | moondream2 Q4 | latest | Smallest capable generative VLM; ingestion-only |
| **Semantic chunker** | semantic-text-splitter | 0.x | Rust-backed, fast, cosine-distance boundary detection |
| **CLI** | click + rich | 8.x / 13.x | Streaming, progress bars, markdown rendering |
| **Config** | tomllib (stdlib) | builtin (3.11+) | Zero dependency |
| **Evaluation** | ragas | 0.2.x | Local LLM judge support |
| **Metadata filtering** | Qdrant payload filters | (via qdrant-client) | Native, no extra library |

---

## 4. Data Models

### 4.1 ChunkMetadata

```python
@dataclass
class ChunkMetadata:
    chunk_id: str           # UUID, primary key
    source_path: str        # Absolute path to source file
    filename: str           # Basename
    source_type: str        # "pdf" | "docx" | "md" | "image" | "audio"

    # Position in document
    char_start: int
    char_end: int
    page_number: Optional[int]     # PDF / DOCX
    section_title: Optional[str]   # Detected section heading

    # Audio-specific
    start_time: Optional[float]    # Seconds
    end_time: Optional[float]      # Seconds

    # Content flags
    has_table: bool = False
    has_image: bool = False
    is_ocr: bool = False
    language: Optional[str] = None

    # Ingestion metadata
    content_hash: str              # SHA-256 of raw text (for dedup)
    indexed_at: str                # ISO 8601 timestamp
    token_count: int
```

### 4.2 Citation

```python
@dataclass
class Citation:
    number: int
    source_type: str
    filepath: str
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    start_time: Optional[float] = None   # Audio: seconds
    end_time: Optional[float] = None     # Audio: seconds
    relevance_score: float = 0.0
    excerpt: str = ""                    # First 150 chars of chunk text

def format_citation(c: Citation) -> str:
    base = f"[{c.number}] {c.filename}"
    if c.source_type == "audio" and c.start_time is not None:
        s = f"{int(c.start_time//60):02d}:{int(c.start_time%60):02d}"
        e = f"{int(c.end_time//60):02d}:{int(c.end_time%60):02d}"
        return f"{base} @ {s}–{e}"
    if c.source_type in ("pdf", "docx"):
        if c.page:    base += f" (p.{c.page})"
        if c.section: base += f" — {c.section}"
    return base
```

### 4.3 Answer

```python
@dataclass
class Answer:
    text: str
    citations: List[Citation]
    confidence: float          # 0.0–1.0 (relevance of top passage)
    used_hyde: bool
    query_latency_ms: int
    retrieval_latency_ms: int
    reranking_latency_ms: int
    generation_latency_ms: int
    tier: str                  # "T1" | "T2" | "T3"
```

---

## 5. Storage Layout

```
~/.ragdb/
├── qdrant/
│   └── collection/
│       ├── 0/
│       │   ├── segments/        # HNSW graph + vector files
│       │   └── wal/             # Write-ahead log
│       └── meta.json
├── bm25/
│   ├── bm25_index.pkl           # rank_bm25 serialized index (small corpus)
│   └── tantivy_index/           # tantivy index directory (large corpus)
├── chunks.db                    # SQLite: chunk text + ChunkMetadata
├── ingestion_tracker.db         # SQLite: filepath → {hash, indexed_at}
└── query_cache.db               # SQLite: query hash → {answer, timestamp}
```

Application config and models are stored separately:
```
./config.toml                   # User config (in project root)
./models/                       # Downloaded model files
    Phi-3.5-mini-instruct-Q4_K_M.gguf
    Qwen2.5-7B-Instruct-Q4_K_M.gguf
    nomic-embed-text-v1.5/      # ONNX model directory
    MiniLM-L12-v2/              # ONNX reranker
    bge-reranker-base/          # ONNX reranker (T3)
    whisper-tiny-q5_k.bin
    whisper-small-q5_k.bin
```

---

## 6. Runtime Memory Budget per Tier

### T1 Query-time (~5.5 GB / 8 GB RAM)
```
OS + Python runtime              2.00 GB
Phi-3.5-mini weights + KV cache  2.28 GB   (2.2 GB + 80 MB KV @ 2048 ctx)
nomic-embed ONNX session         0.55 GB
MiniLM-L6 ONNX session           0.15 GB
Qdrant HNSW graph (on_disk)      0.05 GB
BM25 index (40K chunks)          0.08 GB
SQLite + app                     0.36 GB
─────────────────────────────────────────
Total                           ~5.47 GB   (2.53 GB headroom)
```

### T2 Query-time (VRAM: ~3.1 GB / 4 GB | RAM: ~4.8 GB / 8 GB)
```
VRAM:
  Qwen2.5-7B — 20 GPU layers    2.78 GB
  Token embeddings               0.15 GB
  KV cache (3072 ctx, Q8_0)     0.06 GB
  CUDA overhead                  0.15 GB
  Total VRAM                    ~3.14 GB   (0.86 GB spare)

RAM:
  OS + Python runtime            2.00 GB
  Qwen2.5-7B — 8 CPU layers     1.26 GB
  nomic-embed ONNX               0.55 GB
  MiniLM-L12 ONNX                0.30 GB
  Qdrant + BM25 + SQLite          0.40 GB
  App                            0.30 GB
  Total RAM                     ~4.81 GB   (3.19 GB headroom)
```

### T3 Query-time (VRAM: ~4.5 GB / 6 GB | RAM: ~3.9 GB / 8 GB)
```
VRAM:
  Qwen2.5-7B — 28 GPU layers    4.19 GB
  KV cache (4096 ctx, Q8_0)     0.12 GB
  CUDA overhead                  0.15 GB
  Total VRAM                    ~4.46 GB   (1.54 GB spare on 6 GB)

RAM:
  OS + Python runtime            2.00 GB
  LLM CPU buffers (fully on GPU) 0.20 GB
  nomic-embed ONNX               0.55 GB
  bge-reranker-base ONNX          0.45 GB
  Qdrant + BM25 + SQLite          0.40 GB
  App                            0.30 GB
  Total RAM                     ~3.90 GB   (4.10 GB headroom)
```

---

## 7. Project Directory Structure

```
Motif/                              ← Git repo root
├── cli.py                          ← Main CLI entry point
├── config.toml                     ← User configuration
├── requirements.txt                ← Pinned dependencies
├── setup_models.py                 ← Model download helper
│
├── rag/
│   ├── __init__.py
│   ├── config.py                   ← Config dataclasses + tier detection
│   ├── pipeline.py                 ← End-to-end query orchestration
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parsers/
│   │   │   ├── base.py             ← BasePDFParser, BaseParser ABCs
│   │   │   ├── pdf.py              ← PDF parser (pymupdf + OCR)
│   │   │   ├── docx.py             ← DOCX parser
│   │   │   ├── markdown.py         ← Markdown parser
│   │   │   ├── image.py            ← Image parser (OCR + optional caption)
│   │   │   └── audio.py            ← Audio parser (whisper.cpp)
│   │   ├── chunker.py              ← SemanticChunker / SentenceChunker
│   │   ├── embedder.py             ← nomic-embed ONNX wrapper
│   │   └── deduplicator.py         ← Near-dup detection (SimHash)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py         ← Qdrant local client wrapper
│   │   ├── bm25_index.py           ← rank_bm25 / tantivy wrapper
│   │   ├── fusion.py               ← RRF implementation
│   │   └── expander.py             ← HyDE + multi-query + routing
│   │
│   ├── reranking/
│   │   ├── __init__.py
│   │   └── cross_encoder.py        ← ONNX cross-encoder wrapper
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_client.py           ← llama.cpp wrapper + streaming
│   │   ├── context_builder.py      ← Assembly, ordering, merging, compression
│   │   └── prompts.py              ← All prompt templates
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── chunk_store.py          ← SQLite chunk CRUD
│   │   └── ingestion_tracker.py    ← File hash tracking
│   │
│   ├── models/
│   │   └── model_manager.py        ← Lazy load/unload singleton
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── ragas_runner.py         ← Offline RAGAS evaluation
│       └── test_generator.py       ← Synthetic QA generation
│
├── models/                         ← Downloaded model files (.gguf, ONNX)
│   └── .gitkeep
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── project-context/                ← Engineering documentation
│   ├── context.md
│   ├── architecture.md
│   ├── flow.md
│   ├── trd.md
│   ├── mvp.md
│   ├── instructions.md
│   ├── tests.md
│   └── progress.md
│
├── docs/                           ← Research reports
│   ├── report-1.md
│   ├── report-2 p1.md
│   ├── report-2 p2.md
│   └── report-2 p3.md
│
└── pre_implementation_resolution.md
```
