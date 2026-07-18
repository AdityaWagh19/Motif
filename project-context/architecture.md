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
| `rag/cli.py` | prompt_toolkit REPL entry point; routes plain-text queries and slash commands | `main()`, `_interactive_mode()`, `_handle_slash_command()`, `_handle_query()` |
| `rag/session.py` | Conversation history: list of turns, JSON persist, rolling window trim | `Session`, `Session.add_turn()`, `Session.get_history_for_context()`, `.save()`, `.load()`, `.clear()`, `.new()` |
| `rag/types.py` | **Shared data contracts** — all cross-module dataclasses live here | `Chunk`, `ScoredPassage`, `Citation`, `AnswerResult`, `IngestResult`, `SyncResult` |
| `rag/commands/` | Slash command handlers (one file per command) | `handle_ingest()`, `handle_remove()`, `handle_sync()`, `handle_status()`, `handle_clear()`, `handle_new()`, `handle_setup()`, `handle_help()` |
| `rag/config.py` | Config dataclasses; TOML loading; tier detection | `RAGConfig`, `detect_hardware_tier()`, `load_config()` |
| `rag/pipeline.py` | End-to-end query orchestration — coordinator only, no business logic | `QueryPipeline.answer()` |
| `rag/models/model_manager.py` | Lazy load / unload of all models; single source of model instances | `ModelManager.get_embedder()`, `.get_reranker()`, `.get_llm()`, `.unload()` |
| `rag/models/embedder.py` | nomic-embed-text-v1.5 ONNX INT8 inference wrapper | `Embedder.encode(text)`, `.encode_batch(texts)` |
| `rag/models/reranker.py` | Cross-encoder ONNX inference wrapper (MiniLM / bge-reranker) | `Reranker.score(query, passages)` |
| `rag/ingestion/__init__.py` | **Public ingestion API** — top-level functions consumed by commands | `ingest_path(path, config, recursive, console)`, `remove_document(path, config)`, `sync_directory(path, config, recursive, console)` |
| `rag/ingestion/parsers/` | Modality-specific document parsing | `PDFParser`, `DOCXParser`, `MarkdownParser`, `ImageParser`, `AudioParser` |
| `rag/ingestion/chunker.py` | Sentence / semantic chunking | `SentenceChunker`, `SemanticChunker` |
| `rag/ingestion/deduplicator.py` | Near-duplicate chunk detection via SimHash | `Deduplicator.is_duplicate(chunk)` |
| `rag/retrieval/vector_store.py` | Qdrant HNSW + sparse search wrapper | `VectorStore.search_dense()`, `.search_sparse()`, `.upsert()`, `.delete()` |
| `rag/retrieval/bm25_index.py` | BM25 lexical search | `BM25Index.search()`, `.add()`, `.delete()`, `.rebuild()` |
| `rag/retrieval/fusion.py` | Reciprocal Rank Fusion | `rrf_fuse(dense, sparse, bm25, k=60) -> List[ScoredPassage]` |
| `rag/retrieval/expander.py` | HyDE query expansion + routing heuristic; calls ModelManager for embedder+LLM | `QueryExpander.expand()`, `should_use_hyde(query, config)` |
| `rag/reranking/cross_encoder.py` | Reranking algorithm only — calls `ModelManager.get_reranker()` | `rerank(query, passages, top_k) -> List[ScoredPassage]` |
| `rag/generation/llm_client.py` | llama-cpp-python streaming wrapper | `LLMClient.generate()`, `.stream()` |
| `rag/generation/context_builder.py` | Context assembly: anti-middle ordering, history injection, token budget | `ContextBuilder.build(passages, query, history, config) -> str` |
| `rag/generation/prompts.py` | All prompt templates | `RAG_PROMPT`, `HYDE_PROMPT`, `HISTORY_SYSTEM_PROMPT` |
| `rag/storage/chunk_store.py` | SQLite CRUD for chunk text + ChunkMetadata | `ChunkStore.insert()`, `.fetch()`, `.fetch_batch()`, `.delete_by_source()`, `.count()`, `.count_documents()` |
| `rag/storage/ingestion_tracker.py` | File hash tracking for incremental ingestion | `IngestionTracker.is_indexed()`, `.update()`, `.remove()` |
| `rag/evaluation/ragas_runner.py` | Offline RAGAS evaluation with local LLM judge | `run_evaluation(dataset, metrics)` |
| `rag/evaluation/test_generator.py` | Synthetic QA pair generation from corpus | `create_eval_dataset(chunks, llm, n=100)` |

---

## 3. Dependency Graph

Dependencies are **strictly unidirectional** — no circular imports are possible by design.

```
cli.py (root shim)
  └─► rag.cli

rag.cli
  └─► rag.config, rag.session, rag.commands, rag.pipeline

rag.pipeline
  └─► rag.ingestion, rag.retrieval, rag.reranking, rag.generation,
      rag.storage, rag.models, rag.types

rag.ingestion
  └─► rag.models (embedder via ModelManager), rag.storage, rag.types

rag.retrieval
  └─► rag.models (embedder via ModelManager for HyDE), rag.storage, rag.types

rag.reranking
  └─► rag.models (reranker via ModelManager), rag.types

rag.generation
  └─► rag.models (llm via ModelManager), rag.types

rag.storage
  └─► rag.types

rag.models
  └─► (third-party only: onnxruntime, llama_cpp)

rag.types
  └─► (stdlib only: dataclasses, typing)
```

**Rule:** Nothing may import from `rag.pipeline`, `rag.cli`, or `rag.session`.
All model access goes through `ModelManager` — no module instantiates a model directly.


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

## 4. Data Contracts — `rag/types.py`

All cross-module data types are defined in `rag/types.py`. No other module defines its own result types. This is the single source of truth for data contracts.

```python
# rag/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Chunk:
    """A single indexed unit. Stored in ChunkStore (SQLite) and Qdrant."""
    id: str                          # UUID
    text: str
    source: str                      # Absolute filepath
    filename: str
    source_type: str                 # "pdf" | "docx" | "md" | "image" | "audio"
    page: Optional[int] = None
    section: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    start_time: Optional[float] = None   # Audio (seconds)
    end_time: Optional[float] = None
    has_table: bool = False
    has_image: bool = False
    is_ocr: bool = False
    content_hash: str = ""               # SHA-256, for dedup
    token_count: int = 0
    indexed_at: str = ""                 # ISO 8601

@dataclass
class ScoredPassage:
    """A retrieved chunk with its retrieval score. Passed to reranker and context builder."""
    chunk: Chunk
    score: float                     # RRF score before reranking; reranker score after
    retrieval_method: str            # "dense" | "sparse" | "bm25" | "reranked"

@dataclass
class Citation:
    """A source reference in the answer. Rendered inline as [N]."""
    number: int
    source_type: str
    filepath: str
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    relevance_score: float = 0.0
    excerpt: str = ""               # First ~150 chars of chunk text

@dataclass
class AnswerResult:
    """Returned by QueryPipeline.answer() and consumed by the REPL."""
    text: str
    citations: list[Citation]
    passages_used: int
    used_hyde: bool
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    tier: str

@dataclass
class IngestResult:
    """Returned by rag.ingestion.ingest_path() and consumed by /ingest command."""
    files_processed: int
    chunks_added: int
    files_skipped: int              # Already indexed (dedup / hash unchanged)
    errors: list[str] = field(default_factory=list)

@dataclass
class SyncResult:
    """Returned by rag.ingestion.sync_directory() and consumed by /sync command."""
    added: int
    removed: int
    reindexed: int
    errors: list[str] = field(default_factory=list)
```

> **Rule:** If a function returns data that crosses a module boundary, that return type must be defined in `rag/types.py`. No exceptions.


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

## 8. Project Directory Structure

```
Motif/                              ← Git repo root
├── cli.py                          ← Dev shim (python cli.py → rag.cli:main)
├── pyproject.toml                  ← Package definition; `motif` entry point
├── config.template.toml            ← Fully documented config; copy to config.toml
├── install.sh                      ← Linux/macOS bootstrap installer
├── install.ps1                     ← Windows PowerShell bootstrap installer
├── setup_models.py                 ← Model download helper (`motif setup`)
│
├── rag/                            ← The installable Python package
│   ├── __init__.py                 ← __version__ = "0.1.0"
│   ├── cli.py                      ← prompt_toolkit REPL entry point
│   ├── config.py                   ← RAGConfig dataclasses + tier detection
│   ├── pipeline.py                 ← Query pipeline coordinator
│   ├── session.py                  ← Session: history, JSON persist, /clear, /new
│   ├── types.py                    ← Shared dataclasses: Chunk, ScoredPassage, Citation,
│   │                                  AnswerResult, IngestResult, SyncResult
│   │
│   ├── commands/                   ← Slash command handlers
│   │   ├── __init__.py             ← SLASH_COMMANDS registry + get_command()
│   │   ├── ingest.py               ← /ingest
│   │   ├── remove.py               ← /remove
│   │   ├── sync.py                 ← /sync
│   │   ├── status.py               ← /status
│   │   ├── setup.py                ← /setup (model download)
│   │   ├── clear.py                ← /clear, /new
│   │   └── help.py                 ← /help
│   │
│   ├── models/                     ← Model wrappers ONLY. No pipeline logic.
│   │   ├── __init__.py
│   │   ├── model_manager.py        ← Lazy load/unload singleton
│   │   ├── embedder.py             ← nomic-embed-text-v1.5 ONNX wrapper
│   │   └── reranker.py             ← Cross-encoder ONNX wrapper
│   │
│   ├── ingestion/
│   │   ├── __init__.py             ← PUBLIC API: ingest_path(), remove_document(),
│   │   │                              sync_directory() — consumed by commands layer
│   │   ├── parsers/
│   │   │   ├── base.py             ← BaseParser ABC
│   │   │   ├── pdf.py              ← PyMuPDF + OCR fallback
│   │   │   ├── docx.py             ← DOCX parser
│   │   │   ├── markdown.py         ← Markdown parser
│   │   │   ├── image.py            ← OCR + optional moondream2 caption
│   │   │   └── audio.py            ← whisper.cpp
│   │   ├── chunker.py              ← SentenceChunker / SemanticChunker
│   │   └── deduplicator.py         ← SimHash near-dup detection
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py         ← Qdrant local client wrapper
│   │   ├── bm25_index.py           ← rank_bm25 wrapper
│   │   ├── fusion.py               ← RRF: rrf_fuse() → List[ScoredPassage]
│   │   └── expander.py             ← HyDE + routing heuristic
│   │
│   ├── reranking/
│   │   ├── __init__.py
│   │   └── cross_encoder.py        ← Reranking algorithm (calls ModelManager)
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_client.py           ← llama-cpp-python streaming wrapper
│   │   ├── context_builder.py      ← Assembly, ordering, history injection
│   │   └── prompts.py              ← RAG_PROMPT, HYDE_PROMPT, HISTORY_SYSTEM_PROMPT
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── chunk_store.py          ← SQLite: chunk text + Chunk metadata
│   │   └── ingestion_tracker.py    ← SHA-256 file hash tracking
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── ragas_runner.py         ← Offline RAGAS evaluation
│       └── test_generator.py       ← Synthetic QA generation
│
├── models/                         ← Downloaded .gguf and ONNX files (not committed)
│   └── .gitkeep
│
├── tests/
│   ├── conftest.py                 ← Shared pytest fixtures (tmp Qdrant, SQLite, docs)
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_types.py           ← Chunk, ScoredPassage, AnswerResult construction
│   │   ├── test_chunker.py         ← SentenceChunker token boundaries
│   │   ├── test_bm25.py            ← BM25Index add / search / rebuild
│   │   ├── test_fusion.py          ← RRF score ordering
│   │   ├── test_deduplicator.py    ← SimHash collision rate
│   │   ├── test_citation.py        ← Citation formatting
│   │   └── test_session.py         ← History add/save/load/clear/rolling window
│   └── integration/
│       ├── __init__.py
│       ├── test_ingest_query.py    ← Full ingest → ask round-trip
│       └── test_history.py         ← History persists across Session.save()/load()
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
└── docs/                           ← Research reports
    ├── report-1.md
    ├── report-2 p1.md
    ├── report-2 p2.md
    └── report-2 p3.md
```
