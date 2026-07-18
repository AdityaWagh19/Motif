# Motif

Motif is an offline, multimodal Retrieval-Augmented Generation system for querying local document corpora. It runs entirely on-device — no API keys, no network calls, no cloud dependencies — and adapts its model selection to the available hardware automatically.

---

## Overview

Motif processes documents of multiple types (PDF, DOCX, Markdown, images, audio), indexes them into a hybrid retrieval system (dense vector search + BM25 lexical search), and answers natural-language questions using a local large language model. Every answer is grounded in the indexed documents and includes citations to the source passage, page, or timestamp.

The system is built around three hardware tiers, each with a configuration tuned for accurate answers within the available compute and memory budget.

---

## Hardware Requirements

| Tier | Hardware | RAM | VRAM | LLM | Disk | Faithfulness |
|---|---|---|---|---|---|---|
| T1 | CPU-only | 8 GB | — | Phi-3.5-mini Q4 | 2.8 GB | ~78% |
| T2 | GTX 1650 or equivalent | 8 GB | 4 GB | Qwen2.5-7B Q4_K_M (partial GPU) | 4.9 GB | ~85% |
| T3 | RTX 3050 or equivalent | 8 GB | 6–8 GB | Qwen2.5-7B Q4_K_M (full GPU) | 5.2 GB | ~87% |

The tier is detected automatically at startup. The 5 GB figure covers model weights on disk only. The corpus index scales separately with document volume.

---

## Installation

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/AdityaWagh19/Motif/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/AdityaWagh19/Motif/main/install.ps1 | iex
```

The installer bootstraps `uv`, installs Motif into an isolated environment, detects CUDA and installs the appropriate `llama-cpp-python` wheel automatically, and places `motif` on your PATH. No manual virtual environment or pip invocation required.

After install, download the models for your hardware:
```bash
motif setup           # auto-detect hardware tier, download correct models
motif setup --tier T2 # override tier
```

For development or manual install, see [`project-context/instructions.md`](project-context/instructions.md).

---

## Usage

```bash
# Ingest a folder of documents
python cli.py ingest ./documents/ --recursive

# Check index statistics
python cli.py status

# Ask a question
python cli.py ask "What are the main findings?"

# Restrict retrieval to a specific file
python cli.py ask "Summarize section 3" --file report.pdf

# Restrict to a page range
python cli.py ask "Explain the methodology" --file thesis.pdf --pages 20-40

# Filter by document type
python cli.py ask "What was said about X?" --type audio

# Skip HyDE query expansion (faster)
python cli.py ask "Define gradient descent" --no-hyde

# Remove a document from the index
python cli.py remove ./documents/old_report.pdf

# Sync a folder: add new files, remove deleted files, re-index changed files
python cli.py sync ./documents/
```

---

## Supported Document Types

| Type | Extensions | Notes |
|---|---|---|
| PDF (text) | `.pdf` | All tiers via pymupdf |
| PDF (scanned) | `.pdf` | PaddleOCR on T2; Surya on T3 |
| Word documents | `.docx` | Tables serialized as markdown |
| Markdown | `.md` | Heading hierarchy preserved |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp` | OCR text extraction; optional captioning on T3 |
| Audio | `.mp3`, `.wav`, `.m4a`, `.ogg` | whisper.cpp transcription with timestamps |

---

## Architecture

Motif is structured as a two-phase system:

**Ingestion (one-time):** Documents are parsed by a modality-specific parser, split into semantic chunks (512 tokens, 64-token overlap), embedded with nomic-embed-text-v1.5 (ONNX INT8), and indexed into three complementary stores: a Qdrant HNSW vector index (dense), a Qdrant sparse index, and a BM25 lexical index. Chunk text and metadata are persisted in SQLite.

**Query (per-query):** The query is optionally expanded via HyDE (adaptive, based on query complexity). Results from all three retrieval paths are fused with Reciprocal Rank Fusion (k=60), re-ranked by a cross-encoder (MiniLM-L12 or bge-reranker-base), assembled into a token-budgeted context with anti-lost-in-the-middle ordering, and passed to the local LLM for grounded generation. Output streams to the terminal with source citations.

```
Query
  -> [HyDE expand, T2/T3 adaptive]
  -> nomic-embed encode
  -> Qdrant dense + Qdrant sparse + BM25  ->  RRF fusion  ->  top-20
  -> SQLite fetch (chunk text + metadata)
  -> Cross-encoder rerank  ->  top-3/5
  -> Context assembly (merge adjacent, anti-middle order, extractive compress)
  -> LLM (llama.cpp, streaming)
  -> Answer + Citations
```

Full architecture, data flow diagrams, and interface contracts are documented in [`project-context/`](project-context/).

---

## Project Documentation

| Document | Purpose |
|---|---|
| [`project-context/context.md`](project-context/context.md) | System constraints, hardware tiers, resolved architectural decisions, glossary |
| [`project-context/architecture.md`](project-context/architecture.md) | Component diagram, module responsibilities, tech stack, data models, memory budgets |
| [`project-context/flow.md`](project-context/flow.md) | Ingestion and query pipeline data flows, error handling, interface contracts |
| [`project-context/trd.md`](project-context/trd.md) | Machine-testable acceptance criteria for every subsystem |
| [`project-context/mvp.md`](project-context/mvp.md) | MVP scope, explicit exclusions, acceptance tests, delivery phases |
| [`project-context/instructions.md`](project-context/instructions.md) | Developer setup, model download, configuration, CLI reference, troubleshooting |
| [`project-context/tests.md`](project-context/tests.md) | Unit/integration test specifications, RAGAS evaluation, latency benchmarks |
| [`project-context/progress.md`](project-context/progress.md) | Implementation progress, metrics snapshots, active blockers |
| [`pre_implementation_resolution.md`](pre_implementation_resolution.md) | Gap analysis, hardware tier decisions, redundancy audit |

---

## Technology Stack

| Component | Library |
|---|---|
| LLM inference | llama-cpp-python |
| Embedding model | nomic-embed-text-v1.5 (ONNX INT8) |
| Reranker | MiniLM-L12 / bge-reranker-base (ONNX) |
| Vector store | Qdrant (local, embedded) |
| Lexical index | rank_bm25 / tantivy |
| PDF parsing | pymupdf, PaddleOCR, Surya |
| Audio transcription | whisper.cpp (pywhispercpp) |
| CLI | click, rich |
| Evaluation | RAGAS (offline, local LLM judge) |

---

## Configuration

Copy `config.template.toml` to `config.toml` in the project root. Key settings:

```toml
[hardware]
tier = "auto"        # auto-detect; override: "T1", "T2", "T3"

[llm]
n_gpu_layers = 20    # T1: 0, T2: 20, T3: 28
ctx_size     = 3072  # T1: 2048, T2: 3072, T3: 4096

[retrieval]
query_expansion = "hyde"    # T1: "none"; T2/T3: "hyde" or "none"

[chunking]
use_semantic = true         # T1: false (sentence split), T2/T3: true
```

Full configuration reference is in [`project-context/instructions.md`](project-context/instructions.md).

---

## Research Foundation

The architecture, model selections, and retrieval strategy are derived from a structured literature synthesis covering hybrid retrieval systems, quantized LLM inference, and multimodal document processing. The research reports are in [`docs/`](docs/) and the pre-implementation validation is in [`pre_implementation_resolution.md`](pre_implementation_resolution.md).

---

## Status

Pre-implementation. Documentation and architecture complete. Phase 1 (Foundation) not yet started.

See [`project-context/progress.md`](project-context/progress.md) for current implementation status.

---

## License

To be determined.
