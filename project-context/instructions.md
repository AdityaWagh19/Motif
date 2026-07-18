# Developer Setup & Operating Guide — Motif

> **Depends on:** `architecture.md`, `mvp.md`  
> **Audience:** The developer setting up the project from scratch.

---

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.11+ | Required for `tomllib` builtin |
| Git | Any recent | For version control |
| Disk space | 6 GB free | For models + index |
| RAM | 8 GB | T1 minimum |
| GPU (optional) | CUDA-capable | GTX 1650 (T2) or RTX 3050 (T3) |
| CUDA toolkit | 12.x (if GPU) | Match your driver version |
| OS | Windows / Linux / macOS | Tested on Windows 11 and Ubuntu 22.04 |

---

## 2. Environment Setup

```powershell
# Clone the repo (if not already done)
git clone https://github.com/AdityaWagh19/Motif.git
cd Motif

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2.1 Installing llama-cpp-python with GPU support

llama-cpp-python must be installed with the correct backend for your GPU:

```powershell
# CPU only (T1)
pip install llama-cpp-python

# NVIDIA GPU with CUDA (T2/T3) — Windows
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# NVIDIA GPU with CUDA — Linux
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Verify GPU support:
```python
from llama_cpp import Llama
# Should print: ggml_cuda_init: found N CUDA devices
```

---

## 3. Model Download

Run the download helper script:

```powershell
python setup_models.py --tier T2    # or T1, T3
```

This downloads the correct models for your tier into `./models/`.

### 3.1 Manual Download (if script fails)

**T1 — Phi-3.5-mini Q4_K_M (2.2 GB):**
```powershell
pip install huggingface_hub
huggingface-cli download microsoft/Phi-3.5-mini-instruct-GGUF `
  Phi-3.5-mini-instruct-Q4_K_M.gguf `
  --local-dir models/
```

**T2/T3 — Qwen2.5-7B Q4_K_M (4.2 GB):**
```powershell
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF `
  qwen2.5-7b-instruct-q4_k_m.gguf `
  --local-dir models/
```

**nomic-embed-text-v1.5 ONNX (274 MB):**
```powershell
huggingface-cli download nomic-ai/nomic-embed-text-v1.5-ONNX `
  --local-dir models/nomic-embed-text-v1.5
```

**MiniLM-L12 Reranker ONNX (T1/T2, 134 MB):**
```powershell
# Download via sentence-transformers (will convert to ONNX)
python -c "
from sentence_transformers import CrossEncoder
import onnxruntime as ort
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
model.save('models/MiniLM-L12-v2')
"
```

**bge-reranker-base ONNX (T3 only, 280 MB):**
```powershell
huggingface-cli download BAAI/bge-reranker-base `
  --local-dir models/bge-reranker-base
```

**Whisper tiny (T1/T2, 75 MB):**
```powershell
huggingface-cli download ggerganov/whisper.cpp `
  ggml-tiny-q5_1.bin `
  --local-dir models/
```

**Whisper small (T3, 244 MB):**
```powershell
huggingface-cli download ggerganov/whisper.cpp `
  ggml-small-q5_1.bin `
  --local-dir models/
```

### 3.2 Verify Downloads

```powershell
python setup_models.py --verify
```

Expected output:
```
✓ LLM:      models/Qwen2.5-7B-Instruct-Q4_K_M.gguf  (4.2 GB)
✓ Embed:    models/nomic-embed-text-v1.5/model.onnx  (274 MB)
✓ Reranker: models/MiniLM-L12-v2/model.onnx          (134 MB)
✓ Whisper:  models/ggml-tiny-q5_1.bin                 (75 MB)
All models present. Run: python cli.py status
```

---

## 4. Configuration

Copy the config template and edit for your tier:

```powershell
Copy-Item config.template.toml config.toml
```

### 4.1 Key Settings to Edit

```toml
[hardware]
tier = "auto"   # Change to "T1", "T2", or "T3" to override detection

[models]
llm_path = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

[llm]
# T1: n_gpu_layers = 0, ctx_size = 2048, threads = 4
# T2: n_gpu_layers = 20, ctx_size = 3072, threads = 6
# T3: n_gpu_layers = 28, ctx_size = 4096, threads = 8
n_gpu_layers = 20
ctx_size     = 3072
threads      = 6

[storage]
db_path = "~/.ragdb"   # Change if you want the index elsewhere
```

---

## 5. Quickstart

```powershell
# 1. Ingest a folder of documents
python cli.py ingest ./my_documents/ --recursive

# 2. Check the index
python cli.py status

# 3. Ask a question
python cli.py ask "What are the main conclusions?"

# 4. Ask with source filter
python cli.py ask "What does chapter 3 say about X?" --file report.pdf

# 5. Ask restricting to page range
python cli.py ask "Explain the methodology" --file thesis.pdf --pages 20-40

# 6. Ask without HyDE (faster on all tiers)
python cli.py ask "Define gradient descent" --no-hyde

# 7. Remove a document from the index
python cli.py remove ./my_documents/old_report.pdf

# 8. Sync a folder (add new, remove deleted, re-index changed)
python cli.py sync ./my_documents/
```

---

## 6. CLI Command Reference

| Command | Options | Description |
|---|---|---|
| `ingest PATH [PATH...]` | `--recursive / -r` | Ingest documents into knowledge base |
| `ask QUERY` | `--no-hyde`, `--top-k N`, `--file FILENAME`, `--type TYPE`, `--pages MIN-MAX`, `--consistency`, `--no-sources` | Ask a question |
| `remove PATH` | — | Remove a document and all its chunks |
| `sync DIR` | `--recursive` | Sync directory: add new, remove deleted, re-index changed |
| `status` | — | Show knowledge base stats and loaded models |
| `setup` | `--tier T1/T2/T3`, `--captioning` (T3 only) | Download models for your tier |
| `eval` | `--dataset PATH`, `--output PATH` | Run RAGAS evaluation (Phase 4+) |

---

## 7. Semantic Chunking Note

Semantic chunking (`use_semantic = true`) is enabled on T2/T3 only. It uses nomic-embed to detect semantic boundary changes between sentences. T1 uses sentence-boundary splitting for speed.

Tuning guide for `chunking.semantic_threshold`:
- `0.2` — Tight boundaries (academic/technical docs)
- `0.3` — General documents *(default)*
- `0.4–0.5` — Conversational or narrative text

---

## 8. Troubleshooting Common Failures

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ImportError: No module named llama_cpp` | llama-cpp-python not installed | Run `pip install llama-cpp-python` (or with CUDA flags if GPU) |
| LLM not using GPU | CUDA build not installed | Reinstall with `CMAKE_ARGS="-DGGML_CUDA=on"` |
| Very slow generation (< 5 tok/s) | CPU-only with wrong model | Check `n_gpu_layers` in config; verify CUDA is available |
| High RAM usage during query | LLM partially loaded | Check actual layer split — T2 needs `n_gpu_layers=20` |
| "Cannot find answer" on answerable Q | Retrieval threshold too high | Lower `relevance_threshold` to 0.2 and retry |
| `qdrant_client` write error | Disk full or permissions | Check `~/.ragdb/` disk space; verify write access |
| Audio transcription fails | Wrong whisper model path | Verify path in `config.toml` matches `models/` directory |
| Poor table extraction | Chunker split the table | Ensure `use_semantic = true` or report as bug (tables should be protected) |
| `tomllib` not found | Python < 3.11 | Upgrade Python: `python --version` must be ≥ 3.11 |
| Qdrant HNSW slow on first query | Index warming not done | First query is slow (HNSW loads graph); subsequent queries are fast |
| OCR output is garbled | Low-quality scan | Use Surya (T3) instead of PaddleOCR; set OCR confidence floor |

---

## 9. Advanced Options

### 9.1 moondream2 Image Captioning (T3 opt-in)

```powershell
python cli.py setup --captioning   # Downloads moondream2 Q4 (~900 MB)
```

Then in `config.toml`:
```toml
[parsers]
use_moondream = true
image_density_threshold = 0.3   # 30% image pages triggers captioning
```

Moondream2 is loaded only during ingestion for image-heavy documents, then immediately unloaded.

### 9.2 Encrypted Query Cache (SQLCipher)

For sensitive document sets, replace standard SQLite with SQLCipher:

```powershell
pip install sqlcipher3
```

In `config.toml`:
```toml
[storage]
query_cache_enabled = true
cache_encryption_key = "your-passphrase-here"   # SQLCipher AES-256
```

### 9.3 Large Corpus (>100K chunks) — tantivy BM25

When the BM25 index exceeds 100K chunks, rank_bm25 (in-memory) becomes slow. Switch to tantivy:

```powershell
pip install tantivy
```

In `config.toml`:
```toml
[retrieval]
bm25_backend = "tantivy"   # default: "rank_bm25"
```

The `cli.py sync` command will rebuild the BM25 index in tantivy format on next run.
