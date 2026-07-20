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

### Interactive REPL (primary interface)

```bash
# Launch the interactive session
motif

# Inside the REPL:
/ingest ./documents/          # Ingest a folder of documents
/ingest ./documents/ -r       # Ingest recursively
/status                       # Check index statistics
/sync ./documents/            # Sync: add new, remove deleted, re-index changed
/remove ./documents/old.pdf   # Remove a document
/new                          # Start a fresh session
/help                         # Show all commands

# Ask questions — just type at the prompt:
What are the main findings?

# Inline modifiers to restrict retrieval:
Summarize section 3 /file report.pdf
Explain the methodology /file thesis.pdf /pages 20-40
What was said about X? /type audio
```

### One-shot mode (scripting)

```bash
motif ask "What are the main findings?"
motif ingest ./docs --recursive
```

---

## Supported Document Types

| Type | Extensions | Notes |
|---|---|---|
| PDF (text) | `.pdf` | All tiers via pymupdf |
| PDF (scanned) | `.pdf` | PaddleOCR on T2/T3 |
| Word documents | `.docx` | Tables serialized as markdown |
| Markdown | `.md` | Heading hierarchy preserved |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp` | PaddleOCR text extraction; optional moondream2 captioning on T3 |
| Audio | `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac` | whisper.cpp transcription; WAV must be 16000 Hz |

---

## Architecture

Motif is structured as a two-phase system:

**Ingestion (one-time):** Documents are parsed by a modality-specific parser, split into semantic chunks (T2/T3) or sentence chunks (T1), embedded with nomic-embed-text-v1.5 (ONNX INT8), and indexed into three complementary stores: a Qdrant HNSW vector index (dense), a rank_bm25 lexical index, and a SQLite chunk store.

**Query (per-query):** The query is first classified by an intent classifier (greetings → fast-path; chitchat → LLM without retrieval; document queries → full pipeline). Document queries are optionally expanded via HyDE, then retrieved via hybrid search (dense + BM25), fused with Reciprocal Rank Fusion (k=60), re-ranked by a cross-encoder, assembled into a token-budgeted context, and passed to the local LLM for streaming grounded generation with inline citations.

```
Query
  -> IntentClassifier (embedding cosine similarity)
       -> GREETING_FAST: immediate canned response
       -> CHITCHAT: LLM direct response (no retrieval)
       -> QUERY: full RAG pipeline below
  -> [QueryCache check]
  -> [HyDE expand, T2/T3 adaptive]
  -> nomic-embed encode
  -> Qdrant dense + BM25  ->  RRF fusion  ->  top-N
  -> SQLite fetch (chunk text + metadata)
  -> Cross-encoder rerank  ->  top-3/5
  -> Context assembly (merge adjacent, anti-middle order, extractive compress)
  -> LLM (llama.cpp, streaming via create_chat_completion)
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
| LLM inference | llama-cpp-python (via `create_chat_completion`) |
| Embedding model | nomic-embed-text-v1.5 (ONNX INT8) |
| Reranker | MiniLM-L12-v2 (T1/T2) / bge-reranker-base (T3) ONNX |
| Vector store | Qdrant (local embedded mode, no server) |
| Lexical index | rank_bm25 (auto-upgrades to tantivy >100K chunks) |
| PDF parsing | pymupdf |
| OCR | PaddleOCR (T2/T3) |
| DOCX parsing | python-docx |
| Markdown parsing | markdown-it-py |
| Audio transcription | whisper.cpp (pywhispercpp); requires 16000 Hz WAV |
| Image captioning | moondream2 Q4 (T3 opt-in, ingestion-only) |
| Semantic chunking | semantic-text-splitter (T2/T3) |
| Intent classification | embedding cosine similarity (nomic-embed anchors) |
| CLI / REPL | prompt_toolkit + rich |
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

[storage]
query_cache_enabled = false  # Enable SQLite LRU query cache (500-query limit)
```

Full configuration reference is in [`project-context/instructions.md`](project-context/instructions.md).

---

## Research Foundation

The architecture, model selections, and retrieval strategy are derived from a structured literature synthesis covering hybrid retrieval systems, quantized LLM inference, and multimodal document processing. The research reports are in [`docs/`](docs/) and the pre-implementation validation is in the [gap analysis artifact](pre_implementation_resolution.md).

---

## Status

**Fully implemented.** All six development phases (Infrastructure → Storage → Ingestion → Query Pipeline → Quality & Hardening → Multimodal → Evaluation) are complete. The system is installable globally via `uv tool install`, runs from any directory with the `motif` command, and correctly ingests PDF, DOCX, Markdown, image, and audio documents with grounded cited answers.

See [`project-context/progress.md`](project-context/progress.md) for detailed phase-by-phase status and metrics.

---

## License

MIT
