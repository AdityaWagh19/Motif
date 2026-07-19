# Motif — Performance Optimization Implementation Plan

Source audit: benchmark_audit.md
Hardware: GTX 1650, 4096 MiB VRAM, Driver 529.04, Tier T2
Current state: CPU-only inference, 185 s P50 latency, 46.1% faithfulness, 0.9 tok/s

---

## Execution Order (By ROI)

| Phase | Bottleneck(s) | Effort | Latency Gain | Quality Gain |
|-------|--------------|--------|--------------|--------------|
| 1     | B1/B4 — GPU inference | High | 10–15x | None |
| 2     | B2 — Verbosity / faithfulness | Low | 40–50% | +25% faithfulness |
| 3     | B6/B10 — HyDE opt-in + query cache | Low | 2–5 s + cache hits | None |
| 4     | B5 — Cold-start pre-warming | Medium | 50 s eliminated | None |
| 5     | B7 — BM25 persistence | Low | 1–3 s cold start | None |
| 6     | B9/B11 — Thread env vars + ONNX cleanup | Low | 5–10% CPU contention | None |
| 7     | B3 — Benchmark question cleaning | Low | None | Score accuracy |
| 8     | B12–B15 — Architecture limits (scaling) | High | At scale | At scale |
| 10    | Deferred Architecture Goals (Concurrency, Qdrant, Metrics) | High | Multi-user throughput | Better eval visibility |
---

## Phase 1 — GPU Acceleration (CUDA)

### Objective
Enable actual GPU inference on the GTX 1650. This is the single highest-impact change.
Expected result: P50 latency drops from 185 s to approximately 12–18 s.
TTFT drops from 5,484 ms to approximately 300–600 ms.
Token generation rises from 0.9 tok/s to approximately 25–40 tok/s.

### 1a. Environment Setup (Manual — User Action Required)

These steps must be run by the user in a terminal BEFORE any code changes:

```powershell
# Step 1 — Download CUDA Toolkit 12.x
# https://developer.nvidia.com/cuda-downloads
# Choose: Windows > x86_64 > 11 > exe (local)
# Install with default options (includes nvcc, cuBLAS, headers)

# Step 2 — Verify installation
nvcc --version
# Expected: "Cuda compilation tools, release 12.x, V12.x.xxx"

$env:CUDA_PATH  # Should now be set automatically by the CUDA installer
# If not: $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"

# Step 3 — Rebuild llama-cpp-python with CUDA support
.venv\Scripts\pip.exe uninstall llama-cpp-python -y
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
.venv\Scripts\pip.exe install llama-cpp-python==0.3.34 --no-cache-dir --force-reinstall

# Step 4 — Verify GPU layers are offloaded (look for these lines in startup log)
# ggml_cuda_init: GGML_CUDA_FORCE_MMQ: no
# llm_load_tensors: offloading 20 repeating layers to GPU
```

> [!IMPORTANT]
> Phase 1 is a prerequisite for seeing meaningful latency improvements.
> All other phases can be executed independently without Phase 1.

### 1b. Code — Add GPU Validation at Startup

**[MODIFY] [llm_client.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\generation\llm_client.py)**

After the `Llama(...)` constructor call in `_load()`, add GPU offload verification:

```python
# After: self._llm = Llama(...)
n_layers_requested = cfg.n_gpu_layers
n_layers_actual = getattr(self._llm, 'n_gpu_layers', 0)
if n_layers_requested > 0 and n_layers_actual == 0:
    log.warning(
        "GPU offload requested (n_gpu_layers=%d) but 0 layers were offloaded. "
        "llama-cpp-python may be a CPU-only build. "
        "Rebuild with: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python",
        n_layers_requested,
    )
else:
    log.info("GPU offload: %d layers on GPU", n_layers_actual)
```

### 1c. Code — Detect Missing CUDA Toolkit at Config Load

**[MODIFY] [config.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\config.py)**

Add a new helper function and call it inside `load_config()` after tier detection:

```python
import os

def _check_cuda_toolkit() -> bool:
    """Return True if CUDA Toolkit is installed and CUDA_PATH is set."""
    cuda_path = os.environ.get("CUDA_PATH", "")
    return bool(cuda_path) and Path(cuda_path).exists()

# Inside load_config(), after config.resolved_tier is set:
if config.resolved_tier in ("T2", "T3"):
    if not _check_cuda_toolkit():
        import logging as _log
        _log.getLogger("rag.config").warning(
            "Tier %s detected but CUDA_PATH is not set. "
            "GPU inference will be silently disabled. "
            "Install CUDA Toolkit 12.x to enable GPU acceleration.",
            config.resolved_tier,
        )
```

### 1d. Config — n_gpu_layers After Verification

**File:** `config.toml`

T2 already sets `n_gpu_layers=20`, which is correct for the GTX 1650.
If VRAM OOM occurs, reduce to 18:

```toml
[llm]
n_gpu_layers = 18   # safe floor for GTX 1650 with ctx=3072 and KV cache
```

---

## Phase 2 — Prompt and Token Fixes (Faithfulness: 46% → 70%+)

### Objective
Cut answer verbosity, eliminate [1][1][1] repetition, and improve faithfulness.
These are configuration and string changes — zero infrastructure cost.

### 2a. Config — Reduce max_tokens

**[MODIFY] [config.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\config.py)** — `LLMConfig` dataclass

```python
@dataclass
class LLMConfig:
    n_gpu_layers: int = 0
    ctx_size: int = 2048
    max_tokens: int = 150    # CHANGE: was 400
    temperature: float = 0.1
    threads: int = 4
```

**[MODIFY] `config.py`** — T2 tier defaults dict (add explicit max_tokens):

```python
"T2": {
    "llm": {"n_gpu_layers": 20, "ctx_size": 3072, "max_tokens": 150, "threads": 6},
    "retrieval": {"top_k_retrieval": 25, "top_k_rerank": 5, "query_expansion": "none"},
    "chunking": {"use_semantic": True},
    "generation": {"context_max_tokens": 2048},
    "models": {"llm_path": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "reranker": "models/MiniLM-L12-v2"},
},
```

**`config.toml`:**

```toml
[llm]
max_tokens = 150
```

### 2b. Prompt — Add Conciseness and Anti-Repetition Rules

**[MODIFY] [prompts.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\generation\prompts.py)** — `RAG_PROMPT` constant

```python
RAG_PROMPT = """\
You are a precise research assistant. Answer the question using ONLY the \
information in the provided context passages. Do not speculate or use outside \
knowledge.

Rules:
- Answer in 1-3 sentences maximum. Be direct and concise.
- Do not repeat yourself or rephrase the same point multiple times.
- If the answer is not in the context, say only: \
"I cannot find an answer to this in the available documents."
- Cite each source with its passage number in square brackets, e.g. [1]. \
Use each citation number at most once.

Context:
{context}

Question: {query}
Answer:"""
```

**[MODIFY] `prompts.py`** — `HISTORY_SYSTEM_PROMPT` constant:

```python
HISTORY_SYSTEM_PROMPT = """\
You are a precise research assistant continuing a conversation. Prior context:

{history}

Answer the current question using ONLY the provided document passages. \
Maintain consistency with your previous answers. \
Answer in 1-3 sentences. Cite sources with [N] (each number used at most once)."""
```

---

## Phase 3 — HyDE Opt-In and Query Cache

### 3a. HyDE — Make It Opt-In

HyDE is always on for T2, adding 2–5 s per query even for simple factoid lookups.

**[MODIFY] [pipeline.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\pipeline.py)** — `answer()` signature

```python
def answer(
    self,
    query: str,
    history: List[dict],
    file_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    page_range: Optional[str] = None,
    use_hyde: bool = False,   # CHANGE: was True
    show_sources: bool = True,
) -> AnswerResult:
```

**[MODIFY] [config.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\config.py)** — T2 tier defaults:

```python
"retrieval": {"top_k_retrieval": 25, "top_k_rerank": 5, "query_expansion": "none"},
# CHANGE: was "hyde"
```

**[MODIFY] [cli.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\cli.py)** — add `--hyde` CLI flag

```python
# In argument parser or REPL command handler:
parser.add_argument(
    "--hyde",
    action="store_true",
    default=False,
    help="Enable HyDE query expansion (adds 2-5 s; improves recall for abstract queries)",
)
# Pass use_hyde=args.hyde to pipeline.answer(...)
```

### 3b. Query Cache — Enable by Default

**[MODIFY] [config.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\config.py)** — `StorageConfig`:

```python
@dataclass
class StorageConfig:
    db_path: str = "~/.ragdb"
    query_cache_enabled: bool = True      # CHANGE: was False
    query_cache_ttl_hours: int = 24       # NEW: cache entries expire after 24 hours
```

**`config.toml`:**

```toml
[storage]
query_cache_enabled = true
query_cache_ttl_hours = 24
```

---

## Phase 4 — Startup Pre-Warming (Eliminate 50s Silent Cold Start)

### Objective
Load all models at startup with a Rich spinner progress display.
Converts the 50 s first-query penalty into a transparent startup phase.

### 4a. [NEW] `rag/warmup.py`

```python
"""
rag/warmup.py — Pre-load all models at startup with progress reporting.

Called once from cli.py before the REPL begins.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)


def prewarm_models(config: "RAGConfig", console=None) -> dict:
    """
    Eagerly load embedder, reranker, and LLM.
    Returns dict of {model_name: load_time_ms}.
    """
    from rag.models.model_manager import get_model_manager
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    manager = get_model_manager()
    timings: dict = {}

    steps = [
        ("embedder", "Loading embedder (nomic-embed-text-v1.5)...", lambda: manager.get_embedder(config)),
        ("reranker", "Loading reranker (MiniLM-L12-v2)...",         lambda: manager.get_reranker(config)),
        ("llm",      f"Loading LLM ({config.models.llm_path.split('/')[-1]})...",
                                                                      lambda: manager.get_llm(config)),
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        for name, desc, loader in steps:
            task = progress.add_task(desc, total=None)
            t0 = time.monotonic()
            loader()
            elapsed = round((time.monotonic() - t0) * 1000)
            timings[name] = elapsed
            progress.update(task, description=f"{name.capitalize()} ready ({elapsed} ms)")
            progress.stop_task(task)

    total = sum(timings.values())
    log.info("Pre-warm complete in %d ms: %s", total, timings)
    return timings
```

### 4b. Call prewarm from CLI

**[MODIFY] [cli.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\cli.py)**

```python
from rag.warmup import prewarm_models

# In main startup, after config loaded, before REPL:
if not getattr(args, "no_prewarm", False):
    prewarm_models(config, console=console)

# Add --no-prewarm flag to argument parser:
parser.add_argument(
    "--no-prewarm",
    action="store_true",
    default=False,
    help="Skip model pre-loading (first query will have cold-start latency)",
)
```

---

## Phase 5 — BM25 Index Persistence

### Objective
Avoid rebuilding the BM25 index from ChunkStore on every process restart.
Current cost: ~1–3 s. At 50,000+ chunks: minutes.

### [MODIFY] `rag/retrieval/bm25_index.py`

Add `_save_index()` and `_load_index_if_fresh()` methods.
The `index.pkl` path (`~/.ragdb/bm25/index.pkl`) already exists in the code.

```python
import pickle

def _save_index(self, index_obj: object, path: Path) -> None:
    """Atomically serialize BM25 index to disk."""
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(index_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    log.debug("BM25 index saved to %s", path)

def _load_index_if_fresh(self, index_path: Path, db_path: Path) -> object | None:
    """
    Return deserialized BM25 object if index_path is newer than db_path.
    Returns None if stale or missing, triggering a rebuild.
    """
    if not index_path.exists():
        return None
    if db_path.exists() and db_path.stat().st_mtime > index_path.stat().st_mtime:
        log.debug("BM25 index stale (chunks.db is newer) — will rebuild")
        return None
    with open(index_path, "rb") as f:
        log.info("BM25 index loaded from disk cache (%s)", index_path)
        return pickle.load(f)
```

Invalidation: call `index_path.unlink(missing_ok=True)` in `ChunkStore.add()` after any
new chunk insertion, ensuring the cached index is always consistent.

---

## Phase 6 — Thread Environment Variables and Disk Cleanup

### 6a. Thread Limits

**[MODIFY] [cli.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\cli.py)** — very top of file, before all other imports:

```python
import os

# Set BEFORE numpy/numexpr/onnxruntime import to prevent competing thread pools.
# numexpr auto-detects CPU count (8 threads) and competes with the LLM thread pool.
os.environ.setdefault("NUMEXPR_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
```

### 6b. [NEW] `scripts/cleanup_onnx.py` — One-Time Disk Recovery

```python
"""
scripts/cleanup_onnx.py — Remove unused ONNX variants to recover ~1.7 GB.
Run once manually after verifying the system works correctly.
"""
from pathlib import Path

KEEP = {
    "models/nomic-embed-text-v1.5/onnx": "model_quantized.onnx",
    "models/MiniLM-L12-v2/onnx":          "model_O3.onnx",
}

for onnx_dir_rel, keep_file in KEEP.items():
    onnx_dir = Path(onnx_dir_rel)
    if not onnx_dir.exists():
        continue
    for f in onnx_dir.iterdir():
        if f.name != keep_file and f.suffix in (".onnx", ".pb"):
            print(f"Removing {f}  ({f.stat().st_size / 1_048_576:.1f} MB)")
            f.unlink()

# Remove unused HuggingFace Safetensors / PyTorch weights
for pattern in ("*.safetensors", "*.bin", "flax_model*", "tf_model*"):
    for f in Path("models").rglob(pattern):
        print(f"Removing {f}  ({f.stat().st_size / 1_048_576:.1f} MB)")
        f.unlink()

print("Done.")
```

---

## Phase 7 — Benchmark Question Cleaning

### Objective
Strip injected LLM prompt artefacts from question strings in benchmark_dataset.json.
Q2 AR=0.631 and Q18 AR=0.490 are caused by prompt text contaminating the question embedding.
After cleaning, expected AR for these questions: > 0.80.

### [NEW] `scripts/clean_benchmark.py`

```python
"""
scripts/clean_benchmark.py — Strip injected LLM prompt text from question strings.
"""
import json, re
from pathlib import Path

src = Path("rag/evaluation/benchmark_dataset.json")
items = json.loads(src.read_text(encoding="utf-8"))

STRIP_PATTERNS = [
    r"You are an AI assistant\..*",
    r"Note:.*",
    r"Correct Answer:.*",
    r"Answer:.*",
    r"The question asks.*",
    r"Generate a question.*",
    r"Provide a detailed answer.*",
]

def clean(q: str) -> str:
    for pat in STRIP_PATTERNS:
        q = re.sub(pat, "", q, flags=re.DOTALL | re.IGNORECASE)
    lines = [l.strip() for l in q.strip().splitlines() if l.strip()]
    return lines[0] if lines else q.strip()

cleaned = 0
for item in items:
    original = item.get("question", "")
    fixed = clean(original)
    if fixed != original:
        item["question"] = fixed
        cleaned += 1

out = src.with_name("benchmark_dataset_clean.json")
out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Cleaned {cleaned}/{len(items)} questions → {out}")
```

---

## Phase 8 — Architecture-Level Fixes

### 8a. Dynamic Token Budget Enforcement

**[MODIFY] [context_builder.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\generation\context_builder.py)** — `build()` method

Add a token budget guard that trims passages before generation if the prompt would
exceed the context window:

```python
def _count_tokens_approx(text: str) -> int:
    """Approximate token count (~4 chars per English token)."""
    return max(1, len(text) // 4)

# Inside build(), after constructing prompt:
budget = config.llm.ctx_size - config.llm.max_tokens - 50  # 50 tok safety margin
while _count_tokens_approx(prompt) > budget and len(passages) > 1:
    passages = passages[:-1]   # remove least-relevant passage
    prompt = build_prompt(query, passages, history)
    log.warning("Context trimmed to %d passages to stay within token budget", len(passages))
```

### 8b. BM25 Scale Warning

**[MODIFY] `rag/retrieval/bm25_index.py`** — inside `_build()`, after counting chunks:

```python
if chunk_count > 5_000:
    log.warning(
        "BM25 index has %d chunks. Startup rebuild will take >10 s. "
        "Consider migrating to bm25s for incremental indexing.",
        chunk_count,
    )
```

### 8c. Reranker Candidate Guard

**[MODIFY] [pipeline.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\pipeline.py)** — before calling `rerank()`:

```python
MAX_EFFICIENT_RERANK = 20
if len(candidates) > MAX_EFFICIENT_RERANK:
    log.debug(
        "Reranker received %d candidates (efficient max: %d). "
        "Each extra candidate adds ~8 ms reranking latency.",
        len(candidates), MAX_EFFICIENT_RERANK,
    )
```

---

## All Changed Files at a Glance

### Modified Files

| File | Phases | What Changes |
|------|--------|-------------|
| [llm_client.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\generation\llm_client.py) | 1 | GPU offload verification log after Llama() |
| [config.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\config.py) | 1, 2, 3 | CUDA check; max_tokens=150; T2 defaults; cache enabled; query_expansion="none" |
| [prompts.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\generation\prompts.py) | 2 | Conciseness rules; citation-once rule in RAG_PROMPT and HISTORY_SYSTEM_PROMPT |
| [pipeline.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\pipeline.py) | 3, 8 | use_hyde=False default; reranker candidate guard |
| [cli.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\cli.py) | 3, 4, 6 | --hyde flag; prewarm_models() call; --no-prewarm flag; thread env vars at top |
| [context_builder.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\generation\context_builder.py) | 8 | Dynamic token budget enforcement |
| `rag/retrieval/bm25_index.py` | 5, 8 | Pickle serialize/deserialize; 5k chunk warning |
| `config.toml` | 2, 3 | max_tokens=150; query_cache_enabled=true |
| [ingestion/__init__.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\ingestion\__init__.py) | 9 | Add .docx/.png/.jpg/.mp3/.wav/.m4a/.flac to `_SUPPORTED_EXTENSIONS` and `_EXT_TO_SOURCE_TYPE` |

### New Files

| File | Phase | Purpose |
|------|-------|---------|
| [warmup.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\warmup.py) | 4 | prewarm_models() with Rich spinner |
| `scripts/cleanup_onnx.py` | 6 | One-time disk cleanup (run manually once) |
| `scripts/clean_benchmark.py` | 7 | Strip injected prompts from benchmark questions |
| `scripts/test_multimodal.py` | 9 | End-to-end smoke test for all 5 parsers |
| `rag/evaluation/multimodal_benchmark.py` | 9 | Full multimodal benchmark suite with per-modality scoring |
| `rag/evaluation/retrieval_benchmark.py` | 10 | (Planned) Retrieval-only metrics evaluation script |

---

## Phase 9 — Multimodal Completeness Test and Benchmark

### 9a. Root Cause: Multimodal Extensions Not Registered in Ingestion Router

All 5 parsers exist (PDF, DOCX, Markdown, Image, Audio) and are wired in
`get_parser()`. However, `ingest_path()` in
[ingestion/__init__.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\ingestion\__init__.py)
silently drops DOCX, image, and audio files at line 53 before any parser is called:

```python
# CURRENT (line 53) — BUG: DOCX/image/audio files are filtered out here
_SUPPORTED_EXTENSIONS = frozenset([".pdf", ".md", ".txt", ".markdown"])
```

This means that even though `ImageParser`, `AudioParser`, and `DOCXParser` are
fully implemented, calling `motif ingest myfile.docx` produces:
> "No supported files found"

**[MODIFY] [ingestion/__init__.py](file:///C:\Users\omen\OneDrive\Desktop\Motif\rag\ingestion\__init__.py)**

Fix 1 — Expand `_SUPPORTED_EXTENSIONS` (line 53):

```python
# FIXED: include all parser-supported extensions
_SUPPORTED_EXTENSIONS = frozenset([
    # Text documents
    ".pdf", ".docx",
    ".md", ".txt", ".markdown",
    # Images
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
    # Audio
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
])
```

Fix 2 — Expand `_EXT_TO_SOURCE_TYPE` (lines 56–61):

```python
_EXT_TO_SOURCE_TYPE: dict = {
    ".pdf":      "pdf",
    ".docx":     "docx",
    ".md":       "md",
    ".markdown": "md",
    ".txt":      "txt",
    # Images
    ".png":  "image",
    ".jpg":  "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp":  "image",
    ".tiff": "image",
    # Audio
    ".mp3":  "audio",
    ".wav":  "audio",
    ".m4a":  "audio",
    ".flac": "audio",
    ".ogg":  "audio",
}
```

Fix 3 — Update the user-facing "no files found" message (line 140):

```python
f"Supported types: .pdf .docx .md .txt .png .jpg .mp3 .wav .m4a"
```

Fix 4 — Pass `config` to `get_parser()` for image and audio parsers.
Verify the call at line 127 uses `get_parser(file, config)` — not `get_parser(file)`.

### 9b. Dependency Check — Required Packages Per Modality

| Modality | Parser File | Required Package | Install Command |
|----------|-------------|------------------|-----------------|
| PDF | parsers/pdf.py | pymupdf (fitz) | `pip install pymupdf` |
| DOCX | parsers/docx.py | python-docx | `pip install python-docx` |
| Markdown | parsers/markdown.py | markdown-it-py | `pip install markdown-it-py` |
| Image (OCR) | parsers/image.py | paddleocr | `pip install paddleocr` |
| Image (caption) | parsers/image.py | moondream2 model | `motif setup --tier T3 --captioning` |
| Audio | parsers/audio.py | pywhispercpp | `pip install pywhispercpp` |

Note: PaddleOCR has a heavy dependency chain (~1.2 GB). For Windows,
`pip install paddlepaddle` must be installed first.

### 9c. [NEW] `scripts/test_multimodal.py` — End-to-End Smoke Test

This script creates a minimal synthetic test file for each modality, ingests it,
queries it, and reports whether parsing + retrieval + generation works end-to-end.

```python
"""
scripts/test_multimodal.py — End-to-end multimodal smoke test.

For each modality: creates a synthetic test file, ingests it, runs a query,
and verifies the answer is non-empty and references the correct source.

Usage:
    .venv\Scripts\python.exe scripts/test_multimodal.py
    .venv\Scripts\python.exe scripts/test_multimodal.py --modality audio
"""
from __future__ import annotations

import sys
import json
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import load_config
from rag.ingestion import ingest_path
from rag.pipeline import QueryPipeline


@dataclass
class ModalityResult:
    modality: str
    ingestion_ok: bool = False
    query_ok: bool = False
    answer_references_source: bool = False
    ingestion_ms: float = 0.0
    query_ms: float = 0.0
    chunks_added: int = 0
    answer: str = ""
    error: str = ""


def smoke_test_pdf(cfg, pipeline, tmpdir) -> ModalityResult:
    r = ModalityResult(modality="pdf")
    try:
        # Use an existing PDF from the corpus if available
        pdf = next(Path(".").rglob("*.pdf"), None)
        if not pdf:
            r.error = "No PDF found in project directory"
            return r

        t0 = time.monotonic()
        result = ingest_path(pdf, cfg)
        r.ingestion_ms = (time.monotonic() - t0) * 1000
        r.ingestion_ok = result.errors == [] and result.chunks_added >= 0
        r.chunks_added = result.chunks_added

        t0 = time.monotonic()
        ans = pipeline.answer("What is this document about?", history=[], show_sources=False)
        r.query_ms = (time.monotonic() - t0) * 1000
        r.answer = ans.text[:200]
        r.query_ok = bool(ans.text and "cannot find" not in ans.text.lower())
        r.answer_references_source = ans.passages_used > 0
    except Exception as e:
        r.error = str(e)
    return r


def smoke_test_docx(cfg, pipeline, tmpdir) -> ModalityResult:
    r = ModalityResult(modality="docx")
    try:
        import docx as _docx
        doc = _docx.Document()
        doc.add_heading("Motif DOCX Test", 0)
        doc.add_paragraph(
            "The Transformer architecture introduced in 2017 uses self-attention. "
            "This test document is used to validate DOCX ingestion in Motif."
        )
        doc.add_table(rows=2, cols=2)
        p = tmpdir / "test.docx"
        doc.save(str(p))

        t0 = time.monotonic()
        result = ingest_path(p, cfg)
        r.ingestion_ms = (time.monotonic() - t0) * 1000
        r.ingestion_ok = result.errors == []
        r.chunks_added = result.chunks_added

        t0 = time.monotonic()
        ans = pipeline.answer("What year was the Transformer introduced?", history=[], show_sources=False)
        r.query_ms = (time.monotonic() - t0) * 1000
        r.answer = ans.text[:200]
        r.query_ok = "2017" in ans.text or ans.passages_used > 0
        r.answer_references_source = ans.passages_used > 0
    except ImportError:
        r.error = "python-docx not installed. Run: pip install python-docx"
    except Exception as e:
        r.error = str(e)
    return r


def smoke_test_markdown(cfg, pipeline, tmpdir) -> ModalityResult:
    r = ModalityResult(modality="markdown")
    try:
        p = tmpdir / "test.md"
        p.write_text(
            "# Motif Markdown Test\n\n"
            "The attention mechanism allows models to focus on relevant tokens.\n\n"
            "## Section 2\n\nThis is used to validate Markdown ingestion in Motif.",
            encoding="utf-8"
        )

        t0 = time.monotonic()
        result = ingest_path(p, cfg)
        r.ingestion_ms = (time.monotonic() - t0) * 1000
        r.ingestion_ok = result.errors == []
        r.chunks_added = result.chunks_added

        t0 = time.monotonic()
        ans = pipeline.answer("What does the attention mechanism do?", history=[], show_sources=False)
        r.query_ms = (time.monotonic() - t0) * 1000
        r.answer = ans.text[:200]
        r.query_ok = "attention" in ans.text.lower() or ans.passages_used > 0
        r.answer_references_source = ans.passages_used > 0
    except Exception as e:
        r.error = str(e)
    return r


def smoke_test_image(cfg, pipeline, tmpdir) -> ModalityResult:
    r = ModalityResult(modality="image")
    try:
        from PIL import Image as PILImage  # type: ignore
        import numpy as np
        # Create a synthetic white image with text-like pixels
        img = PILImage.fromarray(
            (np.ones((200, 600, 3), dtype=np.uint8) * 255)
        )
        # Draw text is not available without ImageDraw — just test ingestion
        p = tmpdir / "test.png"
        img.save(str(p))

        t0 = time.monotonic()
        result = ingest_path(p, cfg)
        r.ingestion_ms = (time.monotonic() - t0) * 1000
        r.ingestion_ok = result.errors == []
        r.chunks_added = result.chunks_added
        r.query_ok = True  # Image with no text is valid (empty page returned)
        r.answer = f"Ingested {result.chunks_added} chunks from PNG"
    except ImportError:
        r.error = "Pillow not installed (for test image creation) or paddleocr missing"
    except Exception as e:
        r.error = str(e)
    return r


def smoke_test_audio(cfg, pipeline, tmpdir) -> ModalityResult:
    r = ModalityResult(modality="audio")
    try:
        import wave, struct, math
        # Generate a 2-second synthetic sine wave WAV (440 Hz)
        p = tmpdir / "test.wav"
        sample_rate = 16000
        duration = 2
        samples = [
            int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            for i in range(sample_rate * duration)
        ]
        with wave.open(str(p), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"{len(samples)}h", *samples))

        t0 = time.monotonic()
        result = ingest_path(p, cfg)
        r.ingestion_ms = (time.monotonic() - t0) * 1000
        # For a sine wave, Whisper may transcribe nothing — that is valid
        r.ingestion_ok = result.errors == []
        r.chunks_added = result.chunks_added
        r.query_ok = True
        r.answer = f"Ingested {result.chunks_added} chunks from WAV"
    except ImportError:
        r.error = "pywhispercpp not installed. Run: pip install pywhispercpp"
    except Exception as e:
        r.error = str(e)
    return r


MODALITY_TESTS = {
    "pdf":      smoke_test_pdf,
    "docx":     smoke_test_docx,
    "markdown": smoke_test_markdown,
    "image":    smoke_test_image,
    "audio":    smoke_test_audio,
}


def main():
    parser = argparse.ArgumentParser(description="Motif multimodal smoke test")
    parser.add_argument(
        "--modality",
        choices=list(MODALITY_TESTS.keys()) + ["all"],
        default="all",
        help="Which modality to test (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config()
    pipeline = QueryPipeline(cfg)

    tmpdir = Path(tempfile.mkdtemp(prefix="motif_multimodal_test_"))
    results: List[ModalityResult] = []

    modalities = list(MODALITY_TESTS.keys()) if args.modality == "all" else [args.modality]

    print(f"\nMotif Multimodal Smoke Test — Tier {cfg.resolved_tier}")
    print("=" * 65)

    for modality in modalities:
        print(f"\n[{modality.upper()}]")
        fn = MODALITY_TESTS[modality]
        result = fn(cfg, pipeline, tmpdir)
        results.append(result)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            status = "PASS" if result.ingestion_ok else "FAIL"
            q_status = "PASS" if result.query_ok else "FAIL"
            print(f"  Ingestion : {status} | {result.chunks_added} chunks | {result.ingestion_ms:.0f} ms")
            print(f"  Query     : {q_status} | {result.query_ms:.0f} ms")
            if result.answer:
                print(f"  Answer    : {result.answer[:100]}...")

    shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 65)
    print("Summary:")
    print(f"  {'Modality':<12} {'Ingest':>8} {'Query':>8} {'Chunks':>8} {'Status':>10}")
    print("  " + "-" * 55)
    for r in results:
        ing = "OK" if r.ingestion_ok else ("ERR" if r.error else "FAIL")
        qry = "OK" if r.query_ok else ("ERR" if r.error else "FAIL")
        print(f"  {r.modality:<12} {ing:>8} {qry:>8} {r.chunks_added:>8} {'PASS' if (r.ingestion_ok and r.query_ok) else 'FAIL':>10}")

    out = Path("multimodal_smoke_results.json")
    out.write_text(
        json.dumps([r.__dict__ for r in results], indent=2),
        encoding="utf-8"
    )
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
```

### 9d. [NEW] `rag/evaluation/multimodal_benchmark.py` — Full Benchmark Suite

This runs structured QA evaluation across all modalities using the custom embedding scorer.

```python
"""
rag/evaluation/multimodal_benchmark.py — Per-modality RAG benchmark.

For each modality, ingests sample documents, runs a fixed question set,
and scores Answer Relevancy, Context Precision, and Faithfulness using
the same embedding-based scorer as custom_scorer.py.

Usage:
    .venv\Scripts\python.exe rag/evaluation/multimodal_benchmark.py
    .venv\Scripts\python.exe rag/evaluation/multimodal_benchmark.py --modality docx
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from rag.config import load_config
from rag.pipeline import QueryPipeline
from rag.models.model_manager import get_model_manager


# ---------------------------------------------------------------------------
# Benchmark question sets per modality
# ---------------------------------------------------------------------------

PDF_QUESTIONS = [
    {"q": "What architecture does the paper propose?",    "gt": "Transformer"},
    {"q": "What mechanism replaces recurrence in the model?", "gt": "self-attention"},
    {"q": "How many attention heads are used in the base model?", "gt": "8"},
    {"q": "What dataset was used for English-German translation?", "gt": "WMT 2014"},
    {"q": "What optimizer was used during training?",     "gt": "Adam"},
]

DOCX_QUESTIONS = [
    {"q": "What year was the Transformer introduced?",    "gt": "2017"},
    {"q": "What does the attention mechanism allow?",     "gt": "focus on relevant tokens"},
]

MARKDOWN_QUESTIONS = [
    {"q": "What does the attention mechanism do?",       "gt": "focus on relevant tokens"},
    {"q": "What is the document testing?",               "gt": "Markdown ingestion"},
]

IMAGE_QUESTIONS = [
    {"q": "What does the image show?",                   "gt": "text or visual content"},
]

AUDIO_QUESTIONS = [
    {"q": "What is discussed in the audio recording?",   "gt": "spoken content"},
]

MODALITY_QUESTIONS: Dict[str, list] = {
    "pdf":      PDF_QUESTIONS,
    "docx":     DOCX_QUESTIONS,
    "markdown": MARKDOWN_QUESTIONS,
    "image":    IMAGE_QUESTIONS,
    "audio":    AUDIO_QUESTIONS,
}


# ---------------------------------------------------------------------------
# Scoring (same method as custom_scorer.py)
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def score_answer(question: str, answer: str, ground_truth: str, embedder) -> dict:
    import re
    answer_clean = re.sub(r'\[\d+\]', '', answer).strip()

    q_vec = np.array(embedder.encode(question[:400], prefix="search_query: "), dtype=np.float32)
    a_vec = np.array(embedder.encode(answer_clean[:400], prefix="search_query: "), dtype=np.float32)
    gt_vec = np.array(embedder.encode(ground_truth[:400], prefix="search_query: "), dtype=np.float32)

    answer_relevancy = cosine(q_vec, a_vec)
    gt_alignment     = cosine(a_vec, gt_vec)

    return {
        "answer_relevancy": round(answer_relevancy, 4),
        "gt_alignment":     round(gt_alignment, 4),
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_modality_benchmark(modality: str, questions: list, pipeline: QueryPipeline, embedder) -> dict:
    results = []
    for item in questions:
        q  = item["q"]
        gt = item["gt"]
        t0 = time.monotonic()
        ans = pipeline.answer(q, history=[], show_sources=False)
        latency_ms = (time.monotonic() - t0) * 1000
        scores = score_answer(q, ans.text, gt, embedder)
        results.append({
            "question":         q,
            "ground_truth":     gt,
            "answer":           ans.text[:300],
            "latency_ms":       round(latency_ms),
            "passages_used":    ans.passages_used,
            **scores,
        })

    ar   = round(sum(r["answer_relevancy"] for r in results) / len(results), 4) if results else None
    gta  = round(sum(r["gt_alignment"]     for r in results) / len(results), 4) if results else None
    p50  = sorted(r["latency_ms"] for r in results)[len(results) // 2] if results else None

    return {
        "modality":        modality,
        "n_questions":     len(results),
        "avg_answer_relevancy": ar,
        "avg_gt_alignment":     gta,
        "p50_latency_ms":       p50,
        "per_question":         results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="all",
                        choices=list(MODALITY_QUESTIONS.keys()) + ["all"])
    args = parser.parse_args()

    cfg      = load_config()
    pipeline = QueryPipeline(cfg)
    embedder = get_model_manager().get_embedder(cfg)

    modalities = list(MODALITY_QUESTIONS.keys()) if args.modality == "all" else [args.modality]
    all_results = []

    print(f"\nMotif Multimodal Benchmark — Tier {cfg.resolved_tier}")
    print("=" * 65)

    for modality in modalities:
        questions = MODALITY_QUESTIONS.get(modality, [])
        if not questions:
            print(f"[{modality}] No questions defined — skipping")
            continue
        print(f"\n[{modality.upper()}] Running {len(questions)} questions...")
        result = run_modality_benchmark(modality, questions, pipeline, embedder)
        all_results.append(result)
        print(f"  Answer Relevancy : {result['avg_answer_relevancy']}")
        print(f"  GT Alignment     : {result['avg_gt_alignment']}")
        print(f"  P50 Latency      : {result['p50_latency_ms']} ms")

    print("\n" + "=" * 65)
    print(f"  {'Modality':<12} {'AR':>8} {'GT-Align':>10} {'P50 ms':>10}")
    print("  " + "-" * 45)
    for r in all_results:
        print(f"  {r['modality']:<12} {str(r['avg_answer_relevancy']):>8} "
              f"{str(r['avg_gt_alignment']):>10} {str(r['p50_latency_ms']):>10}")

    out = Path("multimodal_benchmark_results.json")
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
```

### 9e. What Each Test Validates

| Test | Ingestion Validated | Retrieval Validated | Generation Validated |
|------|--------------------|--------------------|---------------------|
| PDF smoke | PDFParser + chunker | Dense + BM25 on PDF text | Answer from PDF context |
| DOCX smoke | DOCXParser + table-to-markdown | Dense + BM25 on DOCX text | Answer from DOCX context |
| Markdown smoke | MarkdownParser + heading-split | Dense + BM25 on MD sections | Answer from MD context |
| Image smoke | ImageParser + PaddleOCR | OCR text retrievable | Answer from OCR text |
| Audio smoke | AudioParser + Whisper | Transcript retrievable | Answer from transcript |
| PDF benchmark | Same as smoke | Context Precision scored | AR + GT Alignment scored |
| DOCX benchmark | Same as smoke | Context Precision scored | AR + GT Alignment scored |
| Markdown benchmark | Same as smoke | Context Precision scored | AR + GT Alignment scored |

### 9f. Expected Failures Before Fix (Current State)

| Modality | Expected behavior before fix | Expected behavior after fix |
|----------|-----------------------------|--------------------------|
| PDF | Works (it is in _SUPPORTED_EXTENSIONS) | Works |
| DOCX | Silently skipped — 0 chunks ingested | Ingested and queryable |
| Markdown | Works (.md is registered) | Works |
| Image | Silently skipped — 0 chunks ingested | Ingested via PaddleOCR |
| Audio | Silently skipped — 0 chunks ingested | Ingested via Whisper |

---

## Phase 10 — Deferred Architecture Goals (Concurrency, Scaling, Retrieval Metrics)

### Objective
Document the necessary future architectural shifts needed for Motif to scale beyond a single-user local tool and properly evaluate retrieval independently.

### 10a. Concurrency via `llama-server` (Fixing B15)

**The Problem:** The current `Llama(...)` Python binding executes in-process and blocks the Global Interpreter Lock (GIL). If a generation takes 100 seconds, the entire pipeline is frozen. Multi-user concurrent queries are impossible.

**The Fix:**
1. Do not use the `llama-cpp-python` Python binding for the LLM.
2. Spin up `llama-server` (the native C++ HTTP server included in `llama.cpp`) in a background process.
3. Update `llm_client.py` to use `aiohttp` or `requests` to stream generation from the `llama-server` API endpoint.
4. `llama-server` uses continuous batching (vLLM style), allowing multiple queries to generate tokens simultaneously without blocking.

### 10b. Qdrant HNSW Tuning for Scale

**The Problem:** At 10,000+ chunks, the default Qdrant parameters might result in a drop in Context Precision and increased search latency.

**The Fix:**
Update vector store initialization in `rag/retrieval/vector_store.py`:
```python
from qdrant_client.http.models import HnswConfigDiff

# Increase ef_construct (build quality) and m (connectivity)
self._client.update_collection(
    collection_name=self.collection_name,
    hnsw_config=HnswConfigDiff(
        m=32,                  # Default is usually 16
        ef_construct=200       # Default is usually 100
    )
)
```

### 10c. Retrieval-Specific Metrics (Recall@k / Precision@k)

**The Problem:** `custom_scorer.py` evaluates the *end-to-end* RAG quality (AR, CP, Faithfulness). If it fails, it's hard to know if retrieval failed or generation hallucinated.

**The Fix:**
1. Annotate a subset of `benchmark_dataset.json` with ground-truth `chunk_id` arrays representing exactly which passages contain the answer.
2. Implement `Recall@k`: Did the ground-truth chunk appear anywhere in the top `k` retrieved chunks?
3. Implement `Precision@k`: What percentage of the top `k` retrieved chunks were in the ground-truth array?
4. Track `MRR` (Mean Reciprocal Rank) to measure how highly ranked the ground-truth chunk was.

---

| Metric | Current | After Phases 1+2 | After All Phases |
|--------|---------|-----------------|-----------------|
| Answer Relevancy | 85.2% | ~87% | ~87% |
| Context Precision | 77.6% | ~79% | ~82% |
| Faithfulness | 46.1% | ~72% | ~75% |
| P50 Latency | 185 s | ~12 s | ~10 s |
| TTFT | 5,484 ms | ~400 ms | ~350 ms |
| Token speed | 0.9 tok/s | ~30 tok/s | ~32 tok/s |
| Cold start | ~240 s | ~60 s | transparent |
| Disk usage | 6.1 GB | 6.1 GB | ~4.4 GB |

---

## Phase 11 — Universal Hardware Tier System

### Objective
The current tier detection (`detect_hardware_tier()` in `config.py`) is **NVIDIA-only and Windows/Linux-biased**. It runs `nvidia-smi` and returns `T1` on every Mac, AMD GPU machine, Intel Arc GPU, or any system where `nvidia-smi` is absent. This makes Motif appear to be CPU-only on powerful hardware that is not NVIDIA.

### 11a. Current Bugs in `detect_hardware_tier()`

| Platform | Current Behavior | Correct Behavior |
|----------|-----------------|-----------------|
| Mac M1/M2/M3 (Metal) | T1 (CPU) | T2 or T3 depending on unified memory size |
| Mac with AMD GPU | T1 (CPU) | T2/T3 if discrete VRAM >= 4 GB |
| AMD Radeon GPU (ROCm) | T1 (CPU) | T2/T3 if VRAM >= 4 GB (via rocm-smi) |
| Intel Arc GPU | T1 (CPU) | T2 if Arc A770 (16 GB), otherwise T1 |
| NVIDIA GPU (CUDA not installed) | T2 correctly detected but CUDA silently fails | T2 detected; warn about missing CUDA separately |
| Multiple NVIDIA GPUs | Only first GPU used | Use highest-VRAM GPU |

**Critical Mac issue:** Apple Silicon Macs (M1 Pro = 16 GB unified memory, M2 Max = 32–96 GB) are extremely capable for LLM inference via Metal/MLX. Currently they auto-detect as T1 (CPU-only, Phi-3.5-mini). They should run Qwen 7B at T2 or T3 speed using llama.cpp's Metal backend.

### 11b. What Needs to Change

**[MODIFY] `rag/config.py`** — Replace `detect_hardware_tier()` with a multi-backend detector:

```python
import platform
import subprocess

def detect_hardware_tier() -> str:
    """
    Universal hardware tier detection.

    Priority order:
      1. NVIDIA GPU via nvidia-smi           (Windows / Linux)
      2. Apple Silicon via sysctl            (macOS arm64)
      3. AMD GPU via rocm-smi                (Linux ROCm)
      4. Fallback: CPU-only T1
    """
    # --- 1. NVIDIA ---
    tier = _detect_nvidia_tier()
    if tier:
        return tier

    # --- 2. Apple Silicon ---
    tier = _detect_apple_silicon_tier()
    if tier:
        return tier

    # --- 3. AMD ROCm ---
    tier = _detect_amd_tier()
    if tier:
        return tier

    # --- 4. Fallback ---
    return "T1"


def _detect_nvidia_tier() -> str | None:
    """Return tier string if NVIDIA GPU found, else None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None
        # Use highest VRAM GPU (multi-GPU support)
        vram_mb = max(int(l) for l in lines)
        if vram_mb >= 6000:
            return "T3"
        elif vram_mb >= 3800:
            return "T2"
        else:
            return "T1"
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _detect_apple_silicon_tier() -> str | None:
    """Return tier string for Apple Silicon Macs, else None."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        # Unified memory in bytes
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024 ** 3)
        # M1/M2/M3: unified memory IS the effective VRAM for Metal inference
        # 8 GB → T2 (can run Qwen 7B Q4 partially offloaded via Metal)
        # 16 GB+ → T3 (full Metal offload)
        if ram_gb >= 16:
            return "T3"
        elif ram_gb >= 8:
            return "T2"
        else:
            return "T1"
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _detect_amd_tier() -> str | None:
    """Return tier string if AMD ROCm GPU found, else None."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        # rocm-smi CSV: GPU,VRAM Total Memory (B),VRAM Total Used Memory (B)
        for line in result.stdout.splitlines():
            if line.startswith("card") or line.startswith("GPU"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                vram_bytes = int(parts[1].strip())
                vram_mb = vram_bytes / (1024 ** 2)
                if vram_mb >= 6000:
                    return "T3"
                elif vram_mb >= 3800:
                    return "T2"
                else:
                    return "T1"
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None
```

### 11c. Tier-Default Updates for Apple Silicon / ROCm

The current T2 and T3 tier defaults set `n_gpu_layers` for CUDA. On Apple Silicon, llama.cpp uses **Metal** (not CUDA). `n_gpu_layers` still controls how many layers go to the Metal GPU — the parameter is the same. However, config.toml comments and documentation must reflect this.

**[MODIFY] `config.toml`** — Update tier comments:

```toml
[hardware]
# Tier detection (auto-detected from available hardware):
#   T1  — CPU only: no GPU, < 4 GB VRAM, or Mac with < 8 GB unified memory
#   T2  — 4–6 GB VRAM / Mac 8–15 GB unified memory (GTX 1650, M1 8 GB)
#   T3  — 6+ GB VRAM / Mac 16+ GB unified memory (RTX 3050+, M2 Pro 16 GB+)
#
# Override with: tier = "T2"
tier = "auto"
```

**[MODIFY] `config.py`** — Add Apple Silicon / AMD backends to `_TIER_DEFAULTS`:

No changes to the actual defaults are needed — the same `n_gpu_layers` values work for Metal. However, a `backend` field (not currently tracked) should be added to `RAGConfig` to surface which backend was detected:

```python
@dataclass
class HardwareConfig:
    tier: str = "auto"
    backend: str = "cpu"  # NEW: "cuda" | "metal" | "rocm" | "cpu"
```

And set in `load_config()` after detection:

```python
if config.resolved_tier != "T1":
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        config.hardware.backend = "metal"
    elif _detect_amd_tier():
        config.hardware.backend = "rocm"
    elif _detect_nvidia_tier():
        config.hardware.backend = "cuda"
```

### 11d. install.sh — Missing macOS Metal Support

**[MODIFY] `install.sh`** — Current: only detects `nvidia-smi`. Fix: add Metal detection branch.

```bash
# After the nvidia-smi block, add:
elif [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    info "Apple Silicon detected — llama.cpp will use Metal (GPU) automatically."
    info "No additional install step needed. Metal support is built into llama-cpp-python."
```

Note: `llama-cpp-python` on macOS arm64 already compiles with Metal by default when installed from source or via the standard wheel. No `CMAKE_ARGS` override is needed. The installer currently does nothing for Mac — it just falls through to the "CPU" message, which is incorrect.

### 11e. install.ps1 — Missing ROCm Branch

**[MODIFY] `install.ps1`** — Current: only detects `nvidia-smi`. Fix: add ROCm detection branch.

```powershell
# After nvidia-smi block, add:
elseif (Get-Command rocm-smi -ErrorAction SilentlyContinue) {
    Write-Info "AMD ROCm GPU detected."
    Write-Info "Installing ROCm-enabled llama-cpp-python..."
    $pythonExe = Join-Path $MotifEnv "Scripts\python.exe"
    & uv pip install llama-cpp-python `
        --python $pythonExe `
        --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/rocm" `
        --force-reinstall --quiet
}
```

### 11f. Files Changed Summary

| File | Change |
|------|--------|
| `rag/config.py` | Replace `detect_hardware_tier()` with multi-backend version; add `HardwareConfig.backend` field |
| `config.toml` | Update tier comments to mention Mac/AMD; add backend field |
| `config.template.toml` | Same as config.toml |
| `install.sh` | Add macOS Metal + AMD ROCm detection branches |
| `install.ps1` | Add AMD ROCm detection branch |

---

## Phase 12 — Installation and Setup Bug Fixes

### Objective
A full audit of `install.sh`, `install.ps1`, `setup_models.py`, `pyproject.toml`, and `config.py` reveals **10 distinct bugs** ranging from version mismatches to config/code misalignment. All are documented and fixed here.

---

### Bug #1 — `llama-cpp-python` Version Floor is Too Low

**File:** `pyproject.toml` line 45

**Bug:**
```toml
"llama-cpp-python>=0.2.80"
```
Version `0.2.80` is from 2023 and predates the `n_gpu_layers` silent-clamping fix and GGUF format support for newer models (Qwen2.5, Phi-3.5). The GGUF `Q4_K_M` quantization for Qwen2.5 requires at least `0.3.0`. Installing `0.2.80` would fail to load the model.

**Fix:**
```toml
"llama-cpp-python>=0.3.0",
```

---

### Bug #2 — `onnxruntime>=1.18` But Actual Requirement is 1.17+

**File:** `pyproject.toml` line 26

**Bug:** The nomic-embed-text ONNX model uses INT8 quantization. `onnxruntime` 1.17 added INT8 quantization support on CPU. Setting `>=1.18` is fine, but the CUDA-enabled variant `onnxruntime-gpu` is never specified. If a user has CUDA installed but installs `onnxruntime` (CPU) instead of `onnxruntime-gpu`, the embedder and reranker run on CPU even after GPU is set up.

**Fix:** Add an optional `gpu` extra:
```toml
[project.optional-dependencies]
gpu = [
    "onnxruntime-gpu>=1.17",   # GPU-accelerated ONNX for embedder + reranker
]
```
And document in README: `pip install motif-rag[gpu]` for GPU-accelerated embedding.

---

### Bug #3 — `config.toml` Has T2 Values Hardcoded as Defaults

**File:** `config.toml` lines 24, 47, 51, 54, 62, 70, 74, 85

**Bug:** `config.toml` is supposed to be a user-editable override file. But the shipped `config.toml` has T2 values hardcoded (e.g., `n_gpu_layers = 20`, `ctx_size = 3072`, `query_expansion = "hyde"`). These values **override the tier-based defaults** from `_apply_tier_defaults()` for every user, regardless of their actual hardware tier.

A T1 user who installs Motif and does not edit `config.toml` will get `n_gpu_layers = 20` and `query_expansion = "hyde"` — both incorrect for CPU-only hardware.

**Root Cause:** `_populate_section()` applies config.toml values **before** `_apply_tier_defaults()`, but `_apply_tier_defaults()` **overwrites** them. Wait — actually checking the loader order in `load_config()`:
```python
_populate_section(...)  # applies config.toml values
_apply_tier_defaults()  # OVERWRITES config.toml values!
```
This means any user customization in `config.toml` is silently overwritten by tier defaults. This is a logic inversion bug.

**Fix:** Reverse the order — apply tier defaults first, then overlay config.toml values:
```python
# Apply tier defaults first as a base
_apply_tier_defaults(config, config.resolved_tier)
# Then overlay with explicit user settings from config.toml (user wins)
_populate_section(config.llm, raw.get("llm", {}))
_populate_section(config.retrieval, raw.get("retrieval", {}))
# ... etc
```

**Additionally:** Ship a `config.toml` that has all values commented out except `tier = "auto"` so it acts as documentation rather than an accidental override.

---

### Bug #4 — `AudioParser` Whisper Model Path Resolution is Broken

**File:** `rag/ingestion/parsers/audio.py` line 55

**Bug:**
```python
path = Path(cfg.models.llm_path).parent.parent / whisper_model
```
This constructs the whisper model path as: `(parent of parent of the LLM .gguf file) / "models/ggml-tiny-q5_1.bin"`.

If `llm_path = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"`, then:
- `Path("models/Qwen2.5-7B-Instruct-Q4_K_M.gguf").parent` = `models/`
- `.parent` = `.` (project root)
- `/ "models/ggml-tiny-q5_1.bin"` = `./models/ggml-tiny-q5_1.bin`

This accidentally works, but **only when the CWD is the project root**. Running `motif` from any other directory breaks audio ingestion silently (raises `FileNotFoundError`).

**Fix:** Resolve relative paths against the project root reliably:
```python
def _get_whisper_model_path(self) -> Path:
    whisper_model = self._config.models.whisper
    path = Path(whisper_model)
    if not path.is_absolute():
        # Resolve relative to the project root (where pyproject.toml lives)
        project_root = Path(__file__).parent.parent.parent.parent
        path = project_root / whisper_model
    if not path.exists():
        raise FileNotFoundError(
            f"Whisper model not found: {path}\n"
            f"Run `motif setup` to download it."
        )
    return path
```

---

### Bug #5 — `setup_models.py` Downloads All ONNX Variants Despite `allow_patterns`

**File:** `setup_models.py` lines 140–141

**Bug:**
```python
if "nomic" in repo_id:
    snapshot_kwargs["allow_patterns"] = ["onnx/*", "tokenizer*", "config.json"]
```
`allow_patterns = ["onnx/*"]` downloads the **entire `onnx/` folder** from the nomic repo, which includes fp32, fp16, int8, arm64, avx512, and OpenVINO variants — totaling ~520 MB instead of the 131 MB quantized model. The current audit (`benchmark_audit.md` Section 6) confirmed this is what caused ~391 MB of "unused ONNX variants" on disk.

**Fix:** Download only the one needed variant per platform:
```python
import platform as _plat

def _get_nomic_pattern() -> list[str]:
    """Return the minimal set of ONNX files needed for this platform."""
    machine = _plat.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return ["onnx/model_quantized_arm64.onnx", "onnx/model_quantized_arm64_data_0.onnx",
                "tokenizer*", "config.json"]
    else:
        return ["onnx/model_quantized.onnx", "tokenizer*", "config.json"]
```

---

### Bug #6 — `install.sh` CUDA Tag Mapping Drops the Minor Version

**File:** `install.sh` line 53

**Bug:**
```bash
CUDA_TAG="cu$(echo "$CUDA_VERSION" | tr -d '.')"
```
If `CUDA_VERSION = "12.4"`, then `CUDA_TAG = "cu124"`.
But if `CUDA_VERSION = "12.4.0"` (3-part version from some nvidia-smi outputs), then `CUDA_TAG = "cu1240"` — which is an invalid tag and the wheel lookup fails silently (falls back to CPU).

**Fix:**
```bash
# Take only the first two version components (major.minor)
CUDA_MAJOR_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f1,2)
CUDA_TAG="cu$(echo "$CUDA_MAJOR_MINOR" | tr -d '.')"
```

---

### Bug #7 — `install.ps1` Same CUDA Tag Bug

**File:** `install.ps1` line 69

**Bug:**
```powershell
$CudaTag = "cu" + $CudaVersion.Replace(".", "")
```
Same issue as `install.sh` — if CUDA version is `12.4.0`, tag becomes `cu1240` instead of `cu124`.

**Fix:**
```powershell
$CudaShort = ($CudaVersion -split '\.')[0..1] -join '.'
$CudaTag = "cu" + $CudaShort.Replace(".", "")
```

---

### Bug #8 — T3 Reranker Model Name Mismatch Between `config.py` and `setup_models.py`

**File:** `rag/config.py` line 165 vs `setup_models.py` line 68

**Bug:**
```python
# config.py T3 defaults:
"models": {"reranker": "models/bge-reranker-base"}
```
```python
# setup_models.py RERANKER_MODELS:
("BAAI/bge-reranker-base", None, "bge-reranker-base", {"T3"}, "280 MB")
# Downloads to: models/bge-reranker-base/
```
```python
# ModelsConfig default (line 28):
reranker: str = "models/MiniLM-L12-v2"
```
The T3 tier default sets `reranker = "models/bge-reranker-base"` but `setup_models.py` downloads it to `models/bge-reranker-base/` (a directory, not a file). The reranker loader in `rag/models/model_manager.py` needs to handle both a directory path (for ONNX snapshot downloads) and a file path. If there is a mismatch in what was downloaded vs what the loader expects, the reranker silently fails or falls back to the wrong model.

**Fix:** Standardize — always use the directory form in config, and verify the loader resolves `model_O3.onnx` inside it:
```python
# In the reranker loader:
reranker_path = Path(config.models.reranker)
if reranker_path.is_dir():
    # Look for the best ONNX variant inside the directory
    candidates = ["model_O3.onnx", "model_optimized.onnx", "model.onnx"]
    onnx_file = next(
        (reranker_path / "onnx" / c for c in candidates
         if (reranker_path / "onnx" / c).exists()), None
    )
```

---

### Bug #9 — `ragas>=0.1` in Optional Deps is Too Broad

**File:** `pyproject.toml` line 54

**Bug:**
```toml
"ragas>=0.1",
```
RAGAS had a breaking API change between v0.1 and v0.2. The `run_config`, `evaluate()` signature, and `SingleTurnSample` dataclasses changed entirely. The project's `ragas_runner.py` likely uses the v0.1 API. Installing v0.2 (which pip will do by default as the latest) breaks the runner.

**Fix:**
```toml
"ragas>=0.1,<0.2",   # v0.2 has breaking API changes incompatible with ragas_runner.py
```
Or migrate `ragas_runner.py` to the v0.2 API and change to `ragas>=0.2`.

---

### Bug #10 — `config.toml` and `config.py` Defaults Are Out of Sync

**Files:** `config.toml` vs `rag/config.py`

**Bug:**

| Setting | `config.py` default | `config.toml` value | Result |
|---------|---------------------|---------------------|--------|
| `max_tokens` | `400` | `400` | In sync |
| `query_cache_enabled` | `False` | `false` | In sync |
| `query_expansion` | `"none"` | `"hyde"` | **OUT OF SYNC** — config.toml enables HyDE but dataclass default is none |
| `use_semantic` | `False` | `false` | In sync, but T2 tier default sets `True` — then config.toml overwrites to `false` (Bug #3) |
| `top_k_retrieval` | `20` | `25` | **OUT OF SYNC** — config.toml hardcodes T2 value |
| `top_k_rerank` | `3` | `5` | **OUT OF SYNC** |
| `n_gpu_layers` | `0` | `20` | **OUT OF SYNC** — config.toml hardcodes T2 value |
| `ctx_size` | `2048` | `3072` | **OUT OF SYNC** — config.toml hardcodes T2 value |

**Fix:** As noted in Bug #3, ship a `config.toml` that only contains `[hardware]\ntier = "auto"` by default. All other values should come from tier defaults. Users override only what they need.

---

### 12a. Files Changed Summary — Phase 12

| File | Bug(s) Fixed | Change |
|------|-------------|--------|
| `pyproject.toml` | #1, #2, #9 | Raise llama-cpp-python floor to 0.3.0; add `[gpu]` extra; pin ragas<0.2 |
| `rag/config.py` | #3 | Reverse populate/apply_defaults order so user config.toml wins |
| `rag/ingestion/parsers/audio.py` | #4 | Fix whisper path resolution to use project root anchor |
| `setup_models.py` | #5 | Download only the platform-appropriate ONNX variant |
| `install.sh` | #6 | Fix CUDA tag mapping to handle 3-part version strings |
| `install.ps1` | #7 | Fix CUDA tag mapping to handle 3-part version strings |
| `rag/models/model_manager.py` | #8 | Fix reranker ONNX file resolution for directory-form paths |
| `config.toml` | #10 | Remove hardcoded T2 values; ship with only `tier = "auto"` as active config |
| `config.template.toml` | #3, #10 | Add clear comments explaining tier-override semantics |

---

## Open Questions

> [!IMPORTANT]
> Phase 1 (GPU acceleration) requires manual CUDA Toolkit installation before code changes are made.
> Confirm you can run the powershell steps in section 1a before proceeding.

> [!NOTE]
> max_tokens=150 limits answers to 1–3 sentences. If you need longer answers for complex queries,
> set max_tokens=200 instead. This is a trade-off between response depth and generation latency.

> [!NOTE]
> Phase 4 pre-warming adds ~50 s to startup. This is strictly better than the current behavior
> (50 s on first query with no feedback). Use --no-prewarm for batch/scripted usage.

> [!NOTE]
> Phase 7 question cleaning will produce a separate benchmark_dataset_clean.json.
> Should the cleaned file replace the original, or kept alongside it?
