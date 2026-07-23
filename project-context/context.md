# Project Context — Motif Offline Multimodal RAG

> **Read this first.** Every other document in `project-context/` depends on the decisions made here.
> Do not change anything in this file without updating downstream documents.

---

## 1. System Objective

Build an **offline, multimodal Retrieval-Augmented Generation (RAG) system** that lets a user ask natural-language questions over a local document corpus and receive grounded, cited answers — entirely without internet access.

**Interface:** CLI-first (`cli.py ask`, `cli.py ingest`, `cli.py sync`, `cli.py status`).  
**Name:** Motif.

---

## 2. Hard Constraints

| Constraint | Value | Scope |
|---|---|---|
| Network access | **None** | All models, data, and computation must be local |
| Model disk footprint | **≤ 5 GB** | Applies to model weight files on disk only (see §2.1) |
| Target accuracy | **≥ 85% RAGAS faithfulness** | On synthetic corpus-specific eval set (T2/T3) |
| Target latency | **Tier-specific** (see §3) | P95 end-to-end query latency |
| Interface | **CLI-first** | Rich terminal output with streaming |
| Python version | **≥ 3.11** | Required for `tomllib` builtin |

### 2.1 — What "5 GB" Means

The 5 GB constraint covers **model weight files on disk** (`.gguf`, ONNX `.onnx`, model parameter directories). It does **not** cover:

- The corpus vector index (scales with content, not application)
- SQLite chunk store (scales with content)
- Application code and config
- OS and Python runtime

This is the only interpretation that is engineering-enforceable at design time. Runtime RAM is tracked separately per hardware tier (§3).

---

## 3. Hardware Tiers

The system detects available hardware and platform backends (NVIDIA CUDA via `nvidia-smi`, Apple Silicon Metal via `sysctl`, AMD ROCm via `rocm-smi`, or CPU fallback) at startup and selects a configuration profile. Users can override via `config.toml`.

### T1 — CPU / 8 GB RAM

| Property | Value |
|---|---|
| Platform Configurations | • Any CPU-only system (x86_64 / arm64)<br>• Intel Macs (x86_64 macOS)<br>• Integrated GPUs (Intel UHD/Iris, AMD APUs)<br>• NVIDIA / AMD GPUs with < 3.8 GB VRAM<br>• Apple Silicon Macs with < 8 GB RAM |
| Backend | CPU (OpenMP / Accelerate) |
| System RAM | 8 GB |
| VRAM | 0 GB |
| LLM | Phi-3.5-mini-instruct Q4_K_M (2.2 GB) |
| GPU layers | 0 / 32 |
| Context Window | 2048 tokens |
| Base Disk footprint | **2.7 GB** |
| Query-time RAM | ~5.5 GB |
| Target faithfulness | ~78% |
| Target latency (P95) | ~11s (no HyDE) |

> T1 does not meet the 85% accuracy target. This is an accepted hardware-constrained trade-off documented explicitly so user expectations are calibrated. T1 is a fully functional system, not a degraded one.

### T2 — GTX 1650 / Apple Silicon 8–15 GB / 4 GB VRAM

| Property | Value |
|---|---|
| Platform Configurations | • NVIDIA GPUs with 3.8–6.0 GB VRAM (GTX 1650, GTX 1060, RTX 3050 4GB)<br>• Apple Silicon M1/M2/M3 with 8–15 GB Unified RAM<br>• AMD Radeon GPUs with 3.8–6.0 GB VRAM (Linux ROCm) |
| Backend | CUDA / Metal / ROCm |
| System RAM | 8 GB |
| VRAM | 3.8–6.0 GB VRAM / 8–15 GB Unified RAM |
| LLM | Qwen2.5-7B-Instruct Q4_K_M (4.2 GB) |
| GPU layers | 20 of 28 (partial offload) |
| Context Window | 3072 tokens |
| Base Disk footprint | **4.7 GB** |
| Query-time VRAM | ~3.14 GB |
| Query-time RAM | ~4.8 GB |
| Target faithfulness | ~85% |
| Target latency (P95) | ~5s (adaptive HyDE) |

### T3 — RTX 3050 / Apple Silicon 16+ GB / 6–8 GB VRAM

| Property | Value |
|---|---|
| Platform Configurations | • NVIDIA GPUs with ≥ 6.0 GB VRAM (RTX 3060/4060/3080/4090, Workstation GPUs)<br>• Apple Silicon M-Series with ≥ 16 GB Unified RAM (M1/M2/M3 Pro/Max/Ultra)<br>• AMD Radeon GPUs with ≥ 6.0 GB VRAM (Linux ROCm, RX 6700+, RX 7800+) |
| Backend | CUDA / Metal / ROCm |
| System RAM | 8+ GB |
| VRAM | ≥ 6.0 GB VRAM / ≥ 16 GB Unified RAM |
| LLM | Qwen2.5-7B-Instruct Q4_K_M (4.2 GB) |
| GPU layers | 28 of 28 (full offload) |
| Context Window | 4096 tokens |
| Base Disk footprint | **5.0 GB** (base; ~5.9 GB with optional moondream2) |
| Query-time VRAM | ~4.46 GB |
| Query-time RAM | ~3.9 GB |
| Target faithfulness | ~87% |
| Target latency (P95) | ~2.5s |

---

## 4. Supported Document Types

| Type | Extensions | Parser | Tier |
|---|---|---|---|
| Text PDF | `.pdf` (digital) | pymupdf | All |
| Scanned PDF | `.pdf` (scanned) | PaddleOCR (T2, T3) | T2, T3 |
| Word documents | `.docx` | python-docx | All |
| Markdown | `.md` | markdown-it-py | All |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp` | PaddleOCR + moondream2 (T3 opt-in) | All |
| Audio | `.mp3`, `.wav`, `.m4a`, `.ogg` | whisper.cpp | All |

---

## 5. Corpus Size Targets

| Scenario | Documents | Chunks (est.) | Index Size |
|---|---|---|---|
| Small (MVP) | 1–50 | up to 10K | < 50 MB |
| Medium (typical) | 50–500 | up to 100K | < 500 MB |
| Large | 500–2000 | up to 500K | < 2 GB |

> BM25 backend switches from `rank_bm25` (in-memory) to `tantivy` (memory-mapped) at the 100K chunk threshold.  
> RAPTOR hierarchical indexing activates only above 500 pages AND with explicit opt-in.

---

## 6. Resolved Architectural Decisions

These decisions are **final**. Revisiting them requires updating this file, `architecture.md`, and `trd.md`.

| Decision | Choice Made | Rejected Alternatives | Rationale |
|---|---|---|---|
| LLM runtime | llama-cpp-python | transformers, vLLM, ollama | Only mature framework for CPU-first quantized offline inference |
| Embedding model | nomic-embed-text-v1.5 ONNX INT8 | BGE-M3, SBERT MiniLM | Best quality/size ratio; 8K context; Matryoshka; ONNX eliminates torch dependency at query time |
| Retrieval strategy | Hybrid: dense + sparse (Qdrant) + BM25, fused via RRF | Dense-only, SPLADE, ColBERT | Dense-only misses exact-match queries; SPLADE requires BGE-M3 (570 MB); ColBERT adds 3–5× index storage for marginal gain after reranking |
| Vector store | Qdrant (local mode, on_disk=True) | FAISS, ChromaDB, Milvus | Native sparse+dense dual-vector support; no server process; HNSW graph in RAM, vectors on disk |
| Sparse index | rank_bm25 → tantivy (>100K chunks) | SPLADE, Elasticsearch | BM25 is sufficient for target corpus sizes; zero additional model weight |
| Reranking | MiniLM-L12 ONNX (T1/T2) / bge-reranker-base ONNX (T3) | UPR (LLM-based), ColBERT | MiniLM-L12: highest ROI per ms; bge-reranker-base adds +2–3% for T3; UPR adds 800ms/query |
| Image captioning | PaddleOCR text-only (base) + moondream2 Q4 (T3 opt-in, ingestion-time only) | CLIP (removed — cannot generate captions), BLIP-2 | CLIP captioning was architecturally incorrect; moondream2 is smallest capable generative VLM |
| Context compression | Extractive (cosine similarity per sentence, using loaded embed model) | LLMLingua-2 | LLMLingua-2 adds 134 MB for ~3% gain; extractive achieves 92% retention at zero extra cost |
| Chunk storage | SQLite (WAL mode, 64 MB cache) | PostgreSQL, MongoDB | Zero-config, ACID, single-file, offline; appropriate for single-user |
| Storage Location | OS-standard `platformdirs` paths with workspace subdirs | Static hardcoded `~/.ragdb` | Adheres to OS platform conventions (`%LOCALAPPDATA%/motif` on Win, `~/.local/share/motif` on Linux); supports multiple named workspaces via `/workspace` command |
| Context window | T1: 2048 / T2: 3072 / T3: 4096 tokens | 1500 / 2048 / 2048 | Qwen2.5-7B handles these safely; more context improves multi-page reasoning |
| HyDE | Adaptive (lightweight heuristic) for T2/T3; Off for T1 | Always on, always off, regex classifier | CPU too slow for HyDE; regex classifier was brittle; adaptive heuristic is robust |
| Query Rewriting | Local LLM `QueryRewriter` (`rag.generation.query_rewriter`) | Raw query string | Rewrites imperative/conversational prompts to keyword search phrases for BM25 and cross-encoder scoring |
| RAPTOR | NumPy k-means clustering + cluster summaries (`rag.ingestion.raptor`) | Heavy scikit-learn dependency | Provides hierarchical thematic summaries over document chunks without extra dependencies |
| FLARE | Dynamic token logprob confidence retrieval (`rag.generation.flare`) | Static context injection | Automatically triggers secondary retrieval if model confidence drops during generation |
| **Primary interface** | **prompt_toolkit REPL** (`motif` launches interactive session) | One-shot CLI only, web UI | Models stay warm between queries; slash commands + plain-text queries in one loop; one-shot mode & flags (`--help`, `--version`) preserved |
| **Workspace isolation** | **`/workspace` command + `<app_dir>/workspaces/<ws>` subdirs** | Single global database | Isolated vector stores and SQLite databases per project/topic |
| **Conversation history** | **Rolling window (last 3 turns), persisted to workspace history** | No history, full history | Single-user local system; history is a list + JSON file; rolling window prevents context budget exhaustion |
| **Installer & CI** | **`uv` + bootstrap scripts + 15-job GitHub Actions CI matrix** | pip, pipx, Homebrew, Docker | `uv` manages Python + venv in one binary; CI workflow enforces 100% pass rate across Linux, Windows, macOS |

---

## 7. Decisions Intentionally Deferred

| Decision | Deferred To | Reason |
|---|---|---|
| HyDE vs multi-query A/B test | After Phase 2 | Needs real corpus for meaningful comparison |
| Optimal relevance threshold | Auto-calibrated on first run | Corpus-dependent; 0.3 is safe default |
| Semantic chunking threshold | 0.3 default, user-tunable | Domain-dependent; no universal optimum |
| Parent-document retrieval | Phase 3 optional | Adds complexity; current stack may be sufficient |
| SQLCipher (encrypted cache) | User opt-in, advanced | Adds key management; not needed for most use cases |
| REST API / GUI | Post-Phase 4 | REPL-first is the primary interface; web UI is a non-goal until pipeline is stable |

---

## 8. Evaluation Targets

| Metric | T1 | T2 | T3 | Method |
|---|---|---|---|---|
| RAGAS Faithfulness | ~78% | ≥85% | ~87% | Synthetic QA from corpus + local LLM judge |
| Answer Relevancy | ~80% | ~88% | ~90% | RAGAS |
| Context Precision | ~82% | ~90% | ~92% | RAGAS |
| Latency P95 (no HyDE) | ~11s | ~5s | ~2.5s | Measured over 100 queries |
| Latency P95 (with HyDE) | N/A | ~8s | ~4.5s | Measured over 100 queries |
| Disk footprint | 2.8 GB | 4.9 GB | 5.2 GB | `du -sh models/` |
| Index build (100 docs) | ≤15 min | ≤10 min | ≤8 min | Timed ingestion run |

Proxy benchmark before real corpus: **FRAMES** (factual + multi-hop QA). Target: ≥75% on FRAMES.

---

## 9. Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — combining document retrieval with LLM generation |
| **Chunk** | A text segment (typically 512 tokens) derived from a source document |
| **Dense retrieval** | Embedding-based similarity search using HNSW index |
| **Sparse retrieval** | Term-frequency-based search (BM25, TF-IDF) |
| **Hybrid retrieval** | Combination of dense + sparse search results via RRF |
| **RRF** | Reciprocal Rank Fusion — score-free rank combination formula |
| **Reranking** | Re-scoring top-K retrieval candidates with a cross-encoder model |
| **HyDE** | Hypothetical Document Embedding — LLM generates a fake answer to improve query embedding |
| **GGUF** | GPT-Generated Unified Format — quantized LLM file format for llama.cpp |
| **ONNX** | Open Neural Network Exchange — portable model format enabling fast CPU inference |
| **GQA** | Grouped-Query Attention — memory-efficient KV cache sharing across attention heads |
| **HNSW** | Hierarchical Navigable Small World — approximate nearest-neighbor graph algorithm |
| **RAGAS** | Retrieval-Augmented Generation Assessment — evaluation framework with local LLM judge |
| **Matryoshka** | Embedding training technique enabling dimension truncation with minimal accuracy loss |
| **T1 / T2 / T3** | Hardware tier designations: CPU-only / GTX 1650 / RTX 3050 |
| **Ingestion** | The process of parsing, chunking, embedding, and indexing source documents |
| **ModelManager** | The singleton responsible for lazy-loading and unloading models across the lifecycle |
   
 