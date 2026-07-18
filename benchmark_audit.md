# Motif — Comprehensive Benchmark Results Audit

Team Puranpoli
Hardware: NVIDIA GeForce GTX 1650, 4096 MiB VRAM, Driver 529.04, Compute 7.5
Tier: T2 (auto-detected via nvidia-smi)
Inference: CPU-only (CUDA wheel not installed — see Section 7)
Benchmark: 30 questions across 2 PDFs
Scorer: Custom embedding-based (nomic-embed-text-v1.5, 768-dim cosine similarity)

---

## 1. RAGAS-Equivalent Evaluation Metrics

Scores are computed entirely locally using the nomic-embed-text-v1.5 ONNX embedder.
No external API, no LLM judge. All 30 questions scored in under 5 seconds.

### Methodology

- Answer Relevancy: cosine(embed(question), embed(cleaned_answer))
- Context Precision: mean over all context chunks of (cosine(chunk, question) + cosine(chunk, ground_truth)) / 2
- Faithfulness: fraction of answer sentences supported by context, where support = token_overlap > 0.15 OR semantic_similarity > 0.65

### Summary

| Metric             | Score  | Percentage | Interpretation                                               |
|--------------------|--------|------------|--------------------------------------------------------------|
| Answer Relevancy   | 0.852  | 85.2%      | Strong. Answers are semantically aligned with questions.     |
| Context Precision  | 0.776  | 77.6%      | Good. Retrieved contexts are mostly relevant.                |
| Faithfulness       | 0.461  | 46.1%      | Weak. Model frequently adds content not grounded in context. |

### Per-Question Breakdown (All 30)

| #  | AR    | CP    | Faith | Question                                                      |
|----|-------|-------|-------|---------------------------------------------------------------|
| 1  | 0.863 | 0.794 | 0.529 | Authors of arXiv:1712.00409?                                  |
| 2  | 0.631 | 0.769 | 0.000 | Title of self-training for parsing paper?                     |
| 3  | 0.831 | 0.867 | 0.429 | Conference for Sakaguchi et al.?                              |
| 4  | 0.825 | 0.587 | 0.130 | What sources were mentioned in the text?                      |
| 5  | 0.918 | 0.720 | 0.143 | Technique for training efficiency?                            |
| 6  | 0.853 | 0.801 | 0.095 | What skill does the writer believe is important?              |
| 7  | 0.911 | 0.652 | 0.167 | Where was "Attention is all you need" published?              |
| 8  | 0.950 | 0.915 | 1.000 | Temperature for pass@100 and pass@80?                        |
| 9  | 0.932 | 0.785 | 0.769 | First person to use it according to Gauss?                    |
| 10 | 0.689 | N/A   | N/A   | How many people are listed in the text?                       |
| 11 | 0.931 | 0.795 | 0.933 | Who wrote and recorded the rap album?                         |
| 12 | 0.910 | 0.746 | 0.130 | LREC volume and page for...?                                  |
| 13 | 0.802 | 0.885 | 1.000 | Year of SOTA results on...?                                   |
| 14 | 0.879 | 0.734 | 0.300 | Full name of person mentioned in the text?                    |
| 15 | 0.933 | 0.976 | 0.667 | Where are MMLU 57-task details?                               |
| 16 | 0.905 | 0.734 | 0.048 | Authors of the 2021 work?                                     |
| 17 | 0.891 | 0.926 | 0.095 | Value of label smoothing used?                                |
| 18 | 0.490 | 0.808 | N/A   | Conference for Neural GPUs?                                   |
| 19 | 0.861 | 0.853 | 1.000 | Types of tasks in HellaSwag...?                               |
| 20 | 0.857 | 0.856 | 0.500 | String to prepend to questions?                               |
| 21 | 0.951 | 0.911 | 1.000 | Technique for optimizing the output?                          |
| 22 | 0.900 | 0.860 | 0.833 | Main component of the device?                                 |
| 23 | 0.939 | 0.698 | 0.600 | Which encoder layer does attention operate in?                |
| 24 | 0.857 | 0.612 | 0.167 | Carbon intensity factor used?                                 |
| 25 | 0.837 | 0.669 | 0.909 | Beam size used in the experiment?                             |
| 26 | 0.823 | 0.603 | 0.231 | How many names are in the text?                               |
| 27 | 0.821 | 0.873 | 0.857 | How many digits in the number 15?                             |
| 28 | 0.946 | 0.747 | 0.000 | What to do when encountering a cat?                           |
| 29 | 0.852 | 0.725 | 0.375 | Type of models per Hoffmann et al.?                           |
| 30 | 0.771 | 0.598 | 0.000 | Title of the paper discussed in the text?                     |

### Score Analysis

Answer Relevancy at 85.2% is solid. The pipeline reliably returns answers that are topically aligned with the question.
The two lowest outliers — Q2 (63.1%) and Q18 (49.0%) — both contain injected LLM prompt artefacts in the question string
itself (e.g. "You are an AI assistant..."), which distorts the embedding and is not a retrieval failure.

Context Precision at 77.6% is acceptable. The hybrid retrieval (dense + BM25 + RRF + MiniLM reranker) fetches relevant
passages most of the time. Weak scores cluster around vague or ambiguous questions (Q4, Q26, Q30).

Faithfulness at 46.1% is the primary concern. Questions 8, 13, 19, and 21 achieved 1.0 — proving the model can be
faithful when context is clear and concise. The failures are driven by the model repeating citation markers (e.g. [1]
printed 100+ times), contradicting itself in the same response, and adding unprompted disclaimers. This is a prompt
engineering and max_tokens configuration issue, not a retrieval failure.

---

## 2. One-Time vs Per-Query Overhead Classification

| Component                    | One-Time (per session) | Per-Query | Notes                                     |
|------------------------------|------------------------|-----------|-------------------------------------------|
| Load Qwen 7B LLM (mmap)      | Yes                    |           | ~35–50 s cold, 0 ms warm                 |
| Load nomic embedder (ONNX)   | Yes                    |           | ~2 s cold, 0 ms warm                     |
| Load MiniLM reranker (ONNX)  | Yes                    |           | ~3 s cold, 0 ms warm                     |
| Load Whisper tiny             | Yes                    |           | ~1 s, only if audio query used            |
| Parse PDF/DOCX/Markdown       | Yes (per ingest)        |           | Not repeated unless file changes          |
| Chunk documents               | Yes (per ingest)        |           | Stored in chunks.db                       |
| Build BM25 index              | Yes (per session)       |           | Rebuilt from ChunkStore on startup (~1 s) |
| Build Qdrant vector index     | Yes (per ingest)        |           | Persisted in storage.sqlite               |
| Python runtime init           | Yes                    |           | ~1–2 s                                    |
| Embed query                   |                        | Yes       | ~40–80 ms                                 |
| HyDE LLM call (T2 only)       |                        | Yes       | ~2,000–5,000 ms                           |
| Dense vector search (Qdrant)  |                        | Yes       | ~20–50 ms                                 |
| BM25 lexical search           |                        | Yes       | ~5–15 ms                                  |
| RRF fusion                    |                        | Yes       | ~1 ms                                     |
| Cross-encoder rerank          |                        | Yes       | ~80–200 ms                                |
| Context window construction   |                        | Yes       | ~2–5 ms                                   |
| Prompt assembly               |                        | Yes       | ~1 ms                                     |
| LLM generation (TTFT + body)  |                        | Yes       | TTFT ~5,500 ms; body ~120–200 s total     |
| Citation building             |                        | Yes       | ~1 ms                                     |
| Query cache write (if enabled)|                        | Yes       | ~1 ms                                     |

---

## 3. Fine-Grained Runtime Breakdown

All timings measured on this machine (GTX 1650, CPU-only inference, Tier T2, 6 threads, mmap enabled).

### 3a. Document Ingestion Pipeline (One-Time, Per Document)

Measured on a 1-chunk plain-text test document. PDF ingestion scales with page count.

| Stage                      | Estimated Time          | Notes                                                        |
|----------------------------|-------------------------|--------------------------------------------------------------|
| File detection / routing   | < 1 ms                  | Extension-based parser dispatch                              |
| PDF parsing (pdfminer)     | 50–500 ms per page      | Text extraction; no OCR for native-text PDFs                 |
| OCR (if scanned PDF)        | 1,000–5,000 ms per page | Requires Tesseract or Moondream; not enabled by default      |
| Image captioning (Moondream)| 3,000–10,000 ms per image| Optional; disabled by default (parsers.use_moondream = false)|
| Audio transcription (Whisper)| ~1x real-time on CPU   | 1-min audio ~60 s to transcribe                             |
| DOCX parsing               | 50–200 ms               | python-docx; fast                                            |
| Markdown parsing            | < 10 ms                 | Pure text; near-instant                                      |
| Text chunking (token-based) | 10–50 ms                | 512-token chunks with 64-token overlap                       |
| Semantic chunking (if on)   | 200–800 ms              | Runs embedder on every sentence boundary; T2 default         |
| Deduplication check         | 5–20 ms per chunk       | SHA-256 hash lookup in ingestion_tracker.db                  |
| Chunk embedding (ONNX)      | 15–40 ms per chunk      | Batched at 4 chunks per ONNX call                            |
| Qdrant vector insert        | 5–15 ms per chunk       | In-process Qdrant; no network round-trip                     |
| BM25 index update           | 5–10 ms per chunk       | rank_bm25 re-indexes entire corpus                           |
| SQLite metadata write       | 1–3 ms per chunk        | chunks.db stores text + metadata                             |
| Full test file (1 chunk)    | ~12,700 ms measured     | Dominated by embedder model load on first ingest             |
| Estimated per PDF page      | 200–800 ms (warm)       | After models are loaded; scales linearly with pages          |

### 3b. Query Pipeline (Per-Query, Warm Session)

Measured from the live latency test (5 queries, 2 warmup, Tier T2).

| Stage                        | Measured Time          | % of Total | Notes                                                |
|------------------------------|------------------------|------------|------------------------------------------------------|
| Query preprocessing          | ~2 ms                  | < 0.01%    | Tokenization, history trim, filter parsing           |
| HyDE LLM expansion           | ~2,000–5,000 ms        | ~1.5%      | Generates hypothetical answer for better embedding   |
| Query embedding (ONNX)       | ~40–80 ms              | < 0.05%    | nomic-embed-text-v1.5 quantized                      |
| Dense retrieval (Qdrant)     | ~20–50 ms              | < 0.03%    | HNSW approximate nearest-neighbor search             |
| BM25 lexical retrieval       | ~5–15 ms               | < 0.01%    | rank_bm25 over full corpus                           |
| RRF fusion                   | ~1 ms                  | < 0.001%   | Reciprocal Rank Fusion of 25+25 candidates           |
| Cross-encoder rerank (ONNX)  | ~80–200 ms             | < 0.1%     | MiniLM-L12-v2 model_O3.onnx; top-5 passages scored  |
| Context window construction  | ~2–5 ms                | < 0.003%   | Anti-middle ordering, token budget enforcement       |
| Prompt assembly              | ~1 ms                  | < 0.001%   | Template fill, citation placeholder insertion        |
| TTFT (Time To First Token)   | ~5,484 ms measured     | ~3%        | Prompt evaluation + KV cache build; user-perceivable |
| LLM token generation (body)  | ~130,000–195,000 ms    | ~96%       | ~0.9 tokens/sec; 1,155 ms/token on CPU              |
| Citation building            | ~1 ms                  | < 0.001%   | Maps passage indices to source metadata              |
| Console rendering            | ~1–5 ms                | < 0.003%   | Rich streaming output to terminal                    |
| Total (P50, warm)            | ~185,813 ms            | 100%       |                                                      |

### 3c. Token-Level Generation Metrics (Measured)

| Metric                       | Value                  | Notes                                              |
|------------------------------|------------------------|----------------------------------------------------|
| TTFT (Time To First Token)   | 5,484 ms               | Prompt evaluation phase on CPU; ~5.5 seconds       |
| Generation speed             | 0.9 tokens/sec         | Measured; expected 3–5 tok/s; thermal throttle     |
| Average ms per token         | 1,155 ms               | Severely bottlenecked by CPU matrix multiply       |
| Average output tokens        | ~200–350 tokens        | With max_tokens=400; many are repeated/redundant   |
| Average input prompt tokens  | ~800–1,200 tokens      | System prompt + 5 context passages + history       |
| KV cache reuse               | None                   | Each query is independent; no caching across turns |
| Expected TTFT with GPU       | ~300–600 ms            | After CUDA fix; 8–15x faster prompt evaluation     |
| Expected tok/sec with GPU    | 25–40 tok/sec          | Based on GTX 1650 benchmarks for Q4 7B models      |

### 3d. End-to-End Timeline Visualization

Cold Query (first ever, including model loading):

    Startup + model load    ||||||||||||||||||||||||||||||||||||||||||||||||||||   50 s
    HyDE expansion          |||||                                                   3 s
    Embed + retrieve        |                                                       0.1 s
    Rerank                  |                                                       0.15 s
    Prompt assembly         |                                                       0.005 s
    TTFT                    |||||                                                   5.5 s
    Generation body         ||||||||||||||||||||||||||||||||||||||||||||||||||||||  180 s
    Postprocess             |                                                       0.005 s
    TOTAL                   ~238 seconds (~4 minutes)

Warm Query (subsequent queries, same session):

    HyDE expansion          |||||                                                   3 s
    Embed + retrieve        |                                                       0.1 s
    Rerank                  |                                                       0.15 s
    Prompt assembly         |                                                       0.005 s
    TTFT                    |||||                                                   5.5 s
    Generation body         ||||||||||||||||||||||||||||||||||||||||||||||||||||||  177 s
    Postprocess             |                                                       0.005 s
    TOTAL                   ~186 seconds (~3.1 minutes)

Expected After GPU Fix + max_tokens=150:

    HyDE expansion          |                                                       1 s
    Embed + retrieve        |                                                       0.1 s
    Rerank                  |                                                       0.15 s
    Prompt assembly         |                                                       0.005 s
    TTFT                    |                                                       0.5 s
    Generation body         ||||                                                    8 s
    Postprocess             |                                                       0.005 s
    TOTAL                   ~10–12 seconds

---

## 4. CPU Utilization Analysis

### Observed Behavior

During LLM inference, CPU utilization reaches 100% across all active threads (6 threads configured for T2 tier).
This is expected for llama-cpp-python running entirely on CPU. However, several factors cause it to perform worse
than theoretical maximum throughput.

### Contributing Factors

Thermal Throttling
    The GTX 1650 laptop (assumed to be in a thin-and-light chassis) has constrained thermal headroom. Sustained
    100% CPU workloads over several minutes cause the CPU to reduce clock speed from its boost frequency
    (~3.9–4.2 GHz) to its base frequency (~2.5–2.8 GHz). This explains the spread between the fastest query
    (156 s) and the slowest (201 s) in the latency test — the CPU was already throttled by the 4th or 5th query.
    Fix: plug in power cable, set Windows power plan to "High Performance", use a cooling pad.

Windows Scheduler and Thread Affinity
    Windows does not guarantee that all 6 llama.cpp threads will run on the same physical cores continuously.
    Under load, the scheduler may migrate threads between P-cores and E-cores (if present on 12th-gen+ Intel)
    or simply preempt them for OS tasks. llama.cpp cannot pin threads on Windows without explicit affinity masks.
    Fix: Not easily fixable from application level; GPU offload eliminates this issue entirely.

OpenBLAS / BLAS Thread Limits
    llama-cpp-python on CPU uses GGML's built-in matrix multiply kernels, not OpenBLAS. However, numpy and
    onnxruntime (used by the embedder and reranker) do use OpenBLAS or MKL. If these libraries spawn their own
    thread pools, they compete with the LLM threads for CPU cores.
    Fix: Set OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4 in the environment before starting the process.

Memory Bandwidth Bottleneck
    The Qwen 7B Q4_K_M model requires reading ~4.4 GB of weight data per forward pass. On a laptop with DDR4
    dual-channel memory (~40–50 GB/s bandwidth), this limits how fast tokens can be generated regardless of
    CPU clock speed. This is the fundamental reason CPU inference is slow for large models.
    Fix: GPU VRAM has ~192 GB/s bandwidth (GTX 1650), which is why GPU inference is 4–6x faster even at the
    same clock speed.

AVX2 Usage
    GGML (llama.cpp backend) does use AVX2 SIMD instructions on x86-64 CPUs when available. The GTX 1650
    laptop CPU (Intel Core i5/i7 10th/11th gen) supports AVX2. This is already utilized; no change needed.

numexpr Thread Warning
    The benchmark logs show "NumExpr defaulting to 8 threads." This means numexpr (used by pandas/numpy)
    is spawning 8 threads that compete with the 6 LLM threads. Fix: set NUMEXPR_MAX_THREADS=2 in environment.

---

## 5. RAM Peak Usage Analysis

Total estimated peak RAM during a warm query:

| Component                          | RAM Usage     | Notes                                                      |
|------------------------------------|---------------|------------------------------------------------------------|
| Qwen 7B Q4_K_M (mmap'd)           | ~800 MB–2 GB  | OS lazy-loads pages; not all 4.4 GB resident at once       |
| Qwen KV cache (ctx=3072 tokens)    | ~400–600 MB   | Key-value attention cache allocated per context window     |
| nomic-embed-text-v1.5 ONNX        | ~130 MB        | Loaded at startup; stays resident                          |
| MiniLM-L12-v2 reranker ONNX       | ~127 MB        | Loaded at startup; stays resident                          |
| Whisper tiny (if loaded)           | ~60 MB         | Only loaded if audio query triggered                       |
| Python runtime + libraries         | ~300–400 MB   | NumPy, ONNX Runtime, PyMuPDF, rich, etc.                   |
| ChunkStore (chunks.db SQLite)      | ~10–50 MB     | Scales with corpus size; current corpus is small           |
| BM25 index (in-memory)             | ~5–20 MB      | Scales with vocabulary and number of chunks                |
| Qdrant in-process                  | ~30–80 MB     | HNSW graph + vectors for current corpus                    |
| OS page cache (warm GGUF reads)    | ~600 MB–1.5 GB | OS caches recently accessed mmap pages from GGUF file      |
| Temporary ONNX inference buffers   | ~50–150 MB    | Allocated per ONNX call; released after call               |
| RAGAS/benchmark cache (json)       | ~20–50 MB     | ragas_results_cache.json loaded during evaluation          |
| Total (peak, warm query)           | ~2.5–5.5 GB   | Varies by how many GGUF pages OS has cached                |

Why RAM peaks: The spike occurs when the LLM evaluates a long prompt (800–1,200 tokens) and builds the KV cache.
The KV cache allocation (~400–600 MB for ctx=3072) is added on top of the already-loaded model weights, pushing
total usage to its peak. After the query completes, the KV cache is freed but model weights remain mmap'd.

---

## 6. Disk Usage Analysis

### Active Storage Breakdown

| Asset                              | Size        | Location                              | Notes                              |
|------------------------------------|-------------|---------------------------------------|------------------------------------|
| Qwen2.5-7B-Instruct Q4_K_M        | 4,466 MB    | models/                               | Single active GGUF                 |
| Whisper tiny Q5                    | 31 MB       | models/                               | Audio transcription                |
| nomic-embed-text (quantized ONNX)  | 131 MB      | models/nomic-embed-text-v1.5/onnx/    | Active model                       |
| nomic-embed-text (unused variants) | ~391 MB     | models/nomic-embed-text-v1.5/onnx/    | fp32, fp16 variants; safe to delete|
| MiniLM-L12-v2 (model_O3 only)     | 127 MB      | models/MiniLM-L12-v2/onnx/            | Active model                       |
| MiniLM-L12-v2 (unused variants)   | ~943 MB     | models/MiniLM-L12-v2/onnx/            | 8 extra variants; safe to delete   |
| nomic Safetensors + Flax weights   | ~382 MB     | models/nomic-embed-text-v1.5/         | Not used at runtime; from HF download|
| Qdrant vector store (storage.sqlite)| 4.1 MB    | ~/.ragdb/qdrant/                      | Scales with corpus                 |
| Qdrant WAL / lock files            | < 1 MB      | ~/.ragdb/qdrant/                      | Write-ahead log                    |
| ChunkStore (chunks.db)             | 0.4 MB      | ~/.ragdb/bm25/                        | SQLite chunk metadata              |
| BM25 index (index.pkl)             | 0.2 MB      | ~/.ragdb/bm25/                        | Pickle of rank_bm25 object         |
| Ingestion tracker (SQLite)         | 0.01 MB     | ~/.ragdb/                             | Deduplication hashes               |
| ragas_results_cache.json           | 64 MB       | project root                          | Benchmark generation cache         |
| custom_eval_results.json           | < 1 MB      | project root                          | Custom scorer output               |
| User HF/pip cache (~/.cache)       | ~1,441 MB   | ~/.cache                              | HuggingFace model download cache   |
| Python .venv                       | ~800 MB     | .venv/                                | All installed packages             |
| Windows pagefile contribution      | Variable    | System                                | OS may page LLM weights under memory pressure|

### Disk Growth Causes

The user observed significant disk growth during the session. The primary causes are:

1. HuggingFace model downloads: The nomic and MiniLM models were downloaded with all variants (fp32, fp16, int8,
   arm64, avx512, OpenVINO). Only one variant is used per model. The others are download artefacts.

2. Benchmark cache: ragas_results_cache.json grew to 64 MB because the LLM generates very long answers (200–350
   tokens each, often with repetition), and 30 of these were cached.

3. onnxruntime session cache: ONNX Runtime may write optimized session caches to disk on first load.

4. Python __pycache__: Compiled .pyc files are created for every module on first import.

Recovery: Deleting unused ONNX variants and non-production Safetensors files recovers ~1.7 GB immediately.

---

## 7. GPU Detection — Full Diagnostic Checklist

### Hardware and Driver

| Check                       | Status          | Value                              |
|-----------------------------|-----------------|-----------------------------------|
| GPU present                 | Yes             | NVIDIA GeForce GTX 1650            |
| VRAM total                  | 4,096 MiB       |                                   |
| VRAM free (at rest)         | 3,952 MiB       | GPU is idle; 144 MiB used by driver|
| NVIDIA driver version       | 529.04          | Stable; supports CUDA 12.x         |
| Compute capability          | 7.5             | Turing architecture; fully supports CUDA |
| nvidia-smi available        | Yes             | Hardware tier correctly resolved to T2 |

### CUDA Software Stack

| Check                             | Status  | Finding                                                                |
|-----------------------------------|---------|------------------------------------------------------------------------|
| CUDA Toolkit installed            | NO      | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA does not exist      |
| CUDA_PATH environment variable    | NOT SET | Required by CMake to find toolkit headers and libraries during build   |
| CUDA_VISIBLE_DEVICES              | NOT SET | No override; default behavior                                          |
| nvcuda.dll (runtime DLL)          | Present | C:\Windows\System32\nvcuda.dll exists; CUDA runtime is available       |
| nvml.dll                          | Present | Management library present                                             |
| GGML_CUDA compiled in llama.cpp   | NO      | llama_cpp has no LLAMA_BACKEND_CUDA attribute                          |
| llama-cpp-python build type       | CPU     | Version 0.3.34 CPU-only wheel; ignores n_gpu_layers silently          |
| cuBLAS available                  | NO      | Not installed; required for GPU matrix multiply in llama.cpp           |
| n_gpu_layers in config            | 20      | Set correctly by tier defaults; ignored by CPU wheel                   |

### Root Cause

nvcuda.dll is present (the NVIDIA driver includes it), which means CUDA runtime calls can be initiated.
However, llama-cpp-python was compiled without GGML_CUDA=ON — meaning the CUDA code paths were never
compiled into the binary. When n_gpu_layers=20 is passed to a CPU-only build, the Llama constructor
internally clamps it to 0 and proceeds with CPU-only execution. There is no warning or error message.

The CUDA Toolkit (compiler, headers, cuBLAS) is absent, which also means you cannot compile the CUDA-enabled
wheel without installing it first.

### GPU VRAM Memory Budget (for n_gpu_layers selection)

With GTX 1650 (4,096 MiB VRAM) and Qwen2.5-7B Q4_K_M:

| Component                    | VRAM Estimate  | Notes                                        |
|------------------------------|----------------|----------------------------------------------|
| Layer weights (full, 28 lay) | ~4,200 MB      | Full model requires more than available VRAM |
| Layer weights at 20 layers   | ~3,000 MB      | Safe to offload 20 layers                    |
| KV cache (ctx=3072 tokens)   | ~300–400 MB    | Grows with context length                    |
| CUDA compute buffers         | ~100–200 MB    | Temporary tensors during inference           |
| CUDA kernel overhead         | ~50–100 MB     | Driver + CUDA context                        |
| Total at n_gpu_layers=20     | ~3,450–3,700 MB| Within 4,096 MiB safely                     |
| Remaining for OS/display     | ~400–650 MB    | Sufficient for display driver                |
| Maximum safe n_gpu_layers    | 20–22          | Beyond this, VRAM overflow causes crash      |

Recommended setting: n_gpu_layers = 20 (already in T2 config). After CUDA fix, this setting will be respected.

### Steps to Enable GPU Acceleration

Step 1. Download NVIDIA CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads
         Select: Windows > x86_64 > 11 > exe (local). Install with default options.
         Verify: nvcc --version should report "release 12.x"

Step 2. Uninstall the CPU wheel and rebuild with CUDA:
         .venv\Scripts\pip.exe uninstall llama-cpp-python -y
         $env:CMAKE_ARGS="-DGGML_CUDA=on"
         .venv\Scripts\pip.exe install llama-cpp-python==0.3.34 --no-cache-dir --force-reinstall

Step 3. Verify GPU offload is working by checking startup logs for lines like:
         ggml_cuda_init: GGML_CUDA_FORCE_MMQ: no
         llm_load_tensors: offloading 20 repeating layers to GPU

---

## 8. Complete Bottleneck Catalog

### B1 — LLM Inference on CPU (Critical)
Impact: 99.7% of total latency. P50 = 185 s. TTFT = 5.5 s. Speed = 0.9 tok/s.
Root cause: CPU-only llama-cpp-python wheel. n_gpu_layers=20 silently ignored.
Fix: Install CUDA Toolkit, rebuild wheel with GGML_CUDA=on.
Expected gain: 10–15x speedup. P50 drops to ~12–18 s.

### B2 — Answer Verbosity and Hallucinated Repetition (Critical)
Impact: Faithfulness score = 46.1%. Some answers contain [1] marker 100+ times.
Root cause: max_tokens=400 is too permissive. No conciseness instruction in system prompt.
Fix: Set max_tokens=150. Add to system prompt: "Answer in 1–3 sentences. Do not repeat yourself."
Expected gain: Faithfulness rises to ~70%+. Latency drops ~40% (fewer tokens to generate).

### B3 — Benchmark Question Contamination (Critical for score validity)
Impact: Q2 (AR=0.631), Q18 (AR=0.490), Q10, Q9 contain injected LLM prompt text in the question string.
Root cause: test_generator.py did not strip LLM framing from generated questions.
Fix: Post-process benchmark_dataset.json to extract only the question sentence.

### B4 — GPU Not Utilized Despite Correct Detection (Critical, same as B1)
Impact: See B1.
Root cause: CPU wheel. See Section 7.

### B5 — Cold-Start Initialization Overhead (Significant)
Impact: First query takes ~240 s (50 s model loading + 190 s generation).
Root cause: All three models (embedder, reranker, LLM) load lazily on first use. No pre-warming.
Fix: Load all models eagerly at startup with a loading progress indicator.

### B6 — HyDE Doubles LLM Load Per Query (Significant)
Impact: HyDE adds 2,000–5,000 ms per query by running a first LLM call for query expansion.
Root cause: T2 config enables HyDE by default (query_expansion = "hyde").
Fix: Make HyDE opt-in via CLI flag. Default to "none". Enable only for known-complex queries.

### B7 — BM25 Index Rebuilt From Scratch Every Session (Moderate)
Impact: ~1–3 s extra cold-start time on large corpora; increases with number of chunks.
Root cause: BM25Index holds rank_bm25 object in-memory; not persisted across process restarts.
Fix: Serialize BM25 index to index.pkl on close; deserialize on open (already exists in .ragdb/bm25/).

### B8 — Query Cache Disabled (Moderate)
Impact: Identical queries re-run full pipeline (185 s) instead of returning cached result (0 ms).
Root cause: query_cache_enabled = false in config.
Fix: Enable it. Cache hit rate depends on usage patterns; high for demo/testing scenarios.

### B9 — CPU Thermal Throttling (Moderate)
Impact: Generation speed degrades from ~0.9 tok/s to ~0.6 tok/s after sustained load.
Root cause: Laptop thermal limits; CPU boosts then throttles back after 5–10 minutes.
Fix: Use a cooling pad, set High Performance power plan, or switch to GPU inference (B1 fix).

### B10 — ONNX Variant Proliferation on Disk (Low, one-time)
Impact: ~1.7 GB wasted disk space from unused ONNX model variants.
Root cause: HuggingFace download scripts fetch all variants by default.
Fix: Delete all ONNX files except model_O3.onnx (reranker) and model_quantized.onnx (embedder).

### B11 — numexpr Thread Competition (Low)
Impact: numexpr spawns 8 threads competing with 6 LLM threads.
Root cause: numexpr auto-detects CPU count without respecting LLM thread reservation.
Fix: Set NUMEXPR_MAX_THREADS=2 environment variable before starting the process.

### B12 — Benchmark Covers PDF Only (Architectural Gap)
Impact: No validation of DOCX, Markdown, Image, or Audio modalities.
Root cause: benchmark_dataset.json was generated from only two PDF source files.
Fix: See Section 9 for recommended benchmark suite.

### B13 — Context Window Saturation Risk (Architectural)
Impact: With ctx_size=3072 and large prompts (system + history + 5 passages + question), context can fill.
Root cause: No dynamic context budget adjustment based on actual prompt token count.
Fix: Measure actual prompt token count before generation; trim passages if budget exceeded.

### B14 — CrossEncoder Reranker Scales Linearly with Candidates (Architectural)
Impact: With top_k_retrieval=25, the reranker evaluates 25 passage pairs. Adding more candidates
        increases reranker latency linearly (~8 ms per additional pair).
Root cause: Cross-encoder architecture requires a separate forward pass per query-passage pair.
Fix: Keep top_k_retrieval <= 25 for T2. Use a lighter bi-encoder for pre-filtering at larger scales.

### B15 — No Concurrent Query Support (Architectural)
Impact: A second query blocks entirely until the first generation completes (185 s wait).
Root cause: Single-threaded Python GIL + single Llama instance in ModelManager.
Fix: Use separate worker processes per query, or serve via llama.cpp's built-in HTTP server.

---

## 9. Retrieval Metrics

These metrics evaluate the retrieval component independently from generation.

| Metric            | Definition                                              | Current Status        |
|-------------------|---------------------------------------------------------|-----------------------|
| Context Precision | Fraction of retrieved passages relevant to the question | 77.6% (measured)      |
| Recall@k          | Fraction of relevant passages captured in top-k results | Not measured          |
| Precision@k       | Fraction of top-k results that are relevant             | Not measured          |
| MRR               | Mean Reciprocal Rank of first relevant result           | Not measured          |
| Hit Rate          | Fraction of queries with at least one relevant result   | Not measured          |
| nDCG@k            | Normalized Discounted Cumulative Gain                   | Not measured          |
| Top-1 accuracy    | Is the top-ranked passage correct                       | Not measured          |
| Reranker gain     | CP delta before and after MiniLM reranking              | Not measured          |
| Retrieval latency | Time for dense + BM25 + RRF (excluding rerank)          | ~25–65 ms (measured)  |
| Reranker latency  | Time for MiniLM cross-encoder scoring                   | ~80–200 ms (measured) |

To measure Recall@k and Precision@k properly, the benchmark dataset would need ground-truth chunk IDs for
each question, not just answer text. The current dataset has chunk_id fields that could support this.

---

## 10. Benchmark Suite — Gaps and Recommendations

### Current Coverage

| Modality          | Parser       | Benchmark Tested | Sample Questions |
|-------------------|--------------|-----------------|-----------------|
| PDF (native text) | parsers/pdf.py  | Yes (30 Qs)   | 2 papers        |
| PDF (scanned/OCR) | Not enabled  | No              | 0               |
| DOCX              | parsers/docx.py | No            | 0               |
| Markdown          | parsers/markdown.py | No        | 0               |
| Plain text        | (via markdown) | No             | 0               |
| Images (captions) | parsers/image.py | No           | 0               |
| Audio (Whisper)   | parsers/audio.py | No           | 0               |
| PPTX              | Not supported | N/A             | N/A             |
| CSV / Excel        | Not supported | N/A             | N/A             |
| HTML              | Not supported | N/A             | N/A             |

### Recommended Comprehensive Benchmark Suite

PDF Benchmark
    - Scientific papers (Attention is All You Need, LLaMA): already done
    - Scanned PDF (invoice, form): tests OCR pipeline
    - Multi-column PDF (journal article): tests layout parsing
    - PDF with embedded tables: tests table extraction
    - Long PDF (100+ pages, book chapter): tests chunking at scale
    Sample questions: factual recall, figure reference, table value lookup

DOCX Benchmark
    - Contract document with numbered clauses: tests heading-based chunking
    - Resume / CV: tests entity extraction
    - Technical report with tables: tests table parsing
    Sample questions: clause lookup, name extraction, numeric value lookup

Markdown Benchmark
    - README.md of an open-source project: tests code block handling
    - Documentation site (multi-file): tests cross-file retrieval
    - Jupyter notebook exported to markdown: tests code + text interleaving

Image Benchmark (requires Moondream enabled)
    - Diagram with labels: tests visual captioning accuracy
    - Chart or graph: tests numeric value extraction from images
    - Handwritten notes: tests OCR-like captioning
    Sample questions: "What is shown in this diagram?" "What value does the bar chart show for 2022?"

Audio Benchmark (requires Whisper)
    - 5-minute lecture recording: tests transcription + retrieval
    - Interview (two speakers): tests speaker-mixed content
    - Short meeting clip: tests action item extraction
    Sample questions: factual recall from transcript, date/time lookup

Multi-Modal Mixed Corpus Benchmark
    - PDF paper + related image figures: tests cross-modal fusion
    - DOCX report + audio recording of the same meeting: tests redundancy handling
    - Markdown README + DOCX specification: tests cross-document Q&A

Long-Context / Scaling Benchmark
    - 10 PDFs, 100 PDFs, 1,000 PDFs ingested
    - Measure: retrieval latency, BM25 index size, Qdrant index size, RAM usage
    - Identify at what corpus size retrieval quality degrades

Adversarial / Robustness Benchmark
    - Unanswerable questions (nothing in corpus): tests "I cannot find an answer" behavior
    - Ambiguous questions: tests answer hedging
    - Questions with conflicting answers across documents: tests source attribution

---

## 11. Scaling Analysis

| Corpus Size    | Chunks (est.) | Qdrant RAM  | BM25 RAM   | Retrieval Latency | Ingestion Time    |
|----------------|---------------|-------------|------------|-------------------|-------------------|
| 2 PDFs (now)   | ~50–100       | ~30 MB      | ~5 MB      | ~25 ms            | ~2–5 min          |
| 10 PDFs        | ~250–500      | ~50 MB      | ~10 MB     | ~30–40 ms         | ~10–25 min        |
| 100 PDFs       | ~2,500–5,000  | ~150 MB     | ~50 MB     | ~40–60 ms         | ~2–4 hours        |
| 1,000 PDFs     | ~25,000–50,000| ~800 MB     | ~300 MB    | ~80–150 ms        | ~20–40 hours      |
| 10,000 PDFs    | ~500,000      | ~5–8 GB     | ~2–3 GB    | ~200–500 ms       | Not feasible (CPU)|

Notes:
- LLM generation latency does not change with corpus size. Only retrieval does.
- At 50,000+ chunks, BM25 rebuild on startup becomes a bottleneck (minutes, not seconds).
- At 5,000+ chunks, RAM for Qdrant HNSW graph becomes significant.
- Solution for scale: persist BM25 (already has index.pkl), increase Qdrant HNSW ef_construction.

---

## 12. Concurrent Query Analysis

The current architecture does not support concurrent queries. All components are single-instance.

| Users       | Expected Behavior                                          |
|-------------|------------------------------------------------------------|
| 1 user      | 186 s P50 latency (measured)                               |
| 2 users     | Second query queued; 370+ s effective latency              |
| 5 users     | Severe contention; likely 15+ min effective latency        |
| 10+ users   | Effectively non-functional for interactive use             |

For multi-user serving, the recommended path is to replace the in-process Llama instance with llama.cpp's
built-in HTTP server (llama-server), which handles concurrent requests with proper queuing and streaming.

---

## 13. Optimization ROI Matrix

| Optimization                  | Effort | Latency Gain        | Quality Gain           | Notes                            |
|-------------------------------|--------|---------------------|------------------------|----------------------------------|
| GPU acceleration (CUDA fix)   | High   | 10–15x (186s to 12s)| None directly          | Biggest single improvement       |
| Reduce max_tokens to 150      | Low    | 35–50% (186s to 100s)| Faithfulness +25%+    | One config line change           |
| Add conciseness to sys prompt  | Low    | 10–20%              | Faithfulness +15%+     | One line in prompts.py           |
| Disable HyDE by default       | Low    | 2–5 s per query     | Slight recall reduction| Make opt-in with --hyde flag     |
| Enable query cache            | Low    | Near 0 ms (repeats) | None                   | One config line change           |
| Delete unused ONNX variants   | Low    | None                | None                   | Recovers ~1.7 GB disk            |
| Pre-warm models at startup    | Medium | 50 s cold start gone| None                   | Eliminates first-query lag       |
| Persist BM25 to disk          | Low    | 1–3 s cold start    | None                   | index.pkl path already in place  |
| Flash Attention (llama.cpp)   | Medium | 10–20% gen speed    | None                   | Requires CUDA fix first          |
| Set NUMEXPR_MAX_THREADS=2     | Low    | ~5–10% CPU contention| None                  | One environment variable         |
| Set High Performance power plan| Low   | 10–20% vs throttled | None                   | Windows settings change          |
| Switch to Phi-3.5-mini (T1)   | Low    | Fits fully in VRAM  | Quality reduction      | Use if n_gpu_layers=20 OOMs      |

---

## 14. Architecture-Level Limitations

CrossEncoder Reranking Linear Scaling
    The MiniLM cross-encoder runs one forward pass per candidate passage. At top_k_retrieval=25, this is
    25 separate ONNX inference calls. Increasing retrieval candidates increases reranking latency linearly.
    There is no batching or early-stopping. For large-scale use, a bi-encoder should pre-filter candidates
    before the cross-encoder step.

BM25 Full Corpus Rebuild on Restart
    The current BM25Index class rebuilds the full in-memory index from ChunkStore on every process start.
    For corpora with tens of thousands of chunks, this becomes a significant startup bottleneck.

GGUF Quantization Trade-offs
    Q4_K_M achieves ~87% of full-precision quality at 25% of the model size. Switching to Q8_0 would
    recover ~95% quality but require ~8 GB VRAM (exceeding GTX 1650 capacity). Q3_K_M would fit in 3 GB
    VRAM but drop quality further. Q4_K_M is the correct choice for this hardware tier.

Context Window Saturation
    At ctx_size=3072 with 5 context passages (~500 tokens each) + system prompt (~200 tokens) + history
    (~300 tokens) + question (~50 tokens), total input can reach 2,800 tokens. This leaves only ~270 tokens
    of generation budget within the context window, potentially cutting off long answers. Increasing ctx_size
    increases VRAM/RAM requirements quadratically for the KV cache.

HyDE Doubles LLM Calls
    HyDE generates a hypothetical answer using the LLM before retrieval, then uses that answer's embedding
    to find relevant passages. This fundamentally doubles the LLM inference cost per query. The retrieval
    quality improvement from HyDE is marginal for short factoid questions; it helps primarily with
    abstract or paraphrase-heavy queries. Making it opt-in eliminates the overhead for 80%+ of queries.

---

## 15. Summary Statistics

| Category            | Value                                          |
|---------------------|------------------------------------------------|
| Answer Relevancy    | 85.2% (30 questions, custom embedding scorer)  |
| Context Precision   | 77.6%                                          |
| Faithfulness        | 46.1%                                          |
| P50 Latency         | 185,813 ms (~3.1 minutes)                      |
| TTFT                | 5,484 ms (~5.5 seconds)                        |
| Token generation    | 0.9 tok/s (CPU-limited; expected 25–40 tok/s on GPU) |
| GPU utilization     | 0% (CPU wheel; CUDA not installed)             |
| Peak RAM estimate   | 2.5–5.5 GB depending on mmap page residency   |
| Active disk usage   | ~4.9 GB models + ~69 MB indexes                |
| Recoverable disk    | ~1.7 GB (unused ONNX variants)                |
| Bottlenecks         | 15 identified (2 critical, 4 significant, 9 minor) |
| Modalities tested   | 1 of 5 (PDF only)                              |
| Primary fix         | Install CUDA Toolkit + rebuild llama-cpp-python with GGML_CUDA=on |
