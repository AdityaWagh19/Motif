# Motif RAG Pipeline — Known Issues & Validation Gaps

This document tracks all currently unresolved problems, performance bottlenecks, and validation gaps blocking the system from being considered a fully functional and optimized multimodal RAG pipeline.

---

## 🔴 Critical — Blocking Correct Functionality

### 1. ONNX Runtime silently running on CPU
This is the biggest unresolved problem. Both the `nomic-embed-text` embedder and the `MiniLM` reranker are falling back to CPU despite the GTX 1650 GPU being available. 
- **Evidence:** 32 chunks took 105 seconds (~3.2s/chunk), which is strictly consistent with CPU-only execution. 
- **Required Fix:** Install `onnxruntime-gpu` and explicitly specify `CUDAExecutionProvider` in the ONNX inference session. Until this is done, we have not achieved true multimodal GPU acceleration — only LLM inference is on the GPU.

### 2. Image pipeline never re-validated after the OCR fix
The `cls=True` / `show_log` PaddleOCR bug is marked **COMPLETED**, but the benchmark was never re-run after fixing it. 
- **Status:** The image modality's 0% retrieval hit rate and 0 chunks indexed are still the last known state. There is not a single validated image ingestion result yet.

### 3. Audio retrieval quality completely unknown
The JFK audio file parsed successfully (1.5s, 1 chunk), but the previous reranker crash meant the retrieval pipeline was never actually exercised for audio. 
- **Status:** Answer rate shows 0%, hit rate is unknown. The ingestion works; whether retrieval works is entirely unproven.

### 4. Zero benchmark coverage for DOCX and Markdown
Both modalities show 0 files, 0 chunks, and no hit rate. 
- **Status:** We cannot claim these are "Active" modalities in any meaningful sense. The benchmark currently fast-fails them in ~30ms and incorrectly calls it done.

---

## 🟠 High Impact — Major Latency/Throughput Problems

### 5. LLM generation is 5–7x slower than theoretical target
- **GPU run:** 72.6s. 
- **Expected:** 10–15s. 
- **Sub-causes:**
  1. `max_tokens` is still set to 400, forcing the model to generate highly verbose and repetitive responses even when a short answer would suffice.
  2. KV cache offloading to VRAM is unconfirmed. If the KV cache is spilling to system RAM, we are losing significant throughput even with CUDA active.

### 6. PDF embedding is catastrophically slow
105 seconds to embed 32 chunks is the direct consequence of Issue #1. Once ONNX GPU is enabled, this should drop to under 2 seconds. Right now, it makes the ingestion pipeline practically unusable for any document of real size.

### 7. No model pre-warming
Models are loaded cold on the very first query, adding a massive, unquantified startup penalty to every session. This makes the first-query latency unreliable and severely inflates benchmark numbers.

---

## 🟡 Validation Gaps — Success Criteria Not Yet Met

### 8. GPU acceleration is only half-verified
Per the success criteria, mathematical verification of GPU use for **both** the LLM and the ONNX models is required.
- We have evidence for `llama.cpp` (2.5x speedup).
- `ONNX` is provably CPU-bound. 
- **Status:** This criterion is explicitly unmet.

### 9. TTFT (Time to First Token) is not measured
The success criteria require TTFT below 1.5 seconds. The current benchmark logs do not isolate TTFT from total generation time. We are measuring end-to-end latency (72.6s), not prefill time, making it impossible to evaluate this criterion accurately.

### 10. No RAGAS metrics for the GPU run
Answer Relevancy and Context Precision were not calculated because the cache was cleared and the reranker bug interrupted evaluations. There is currently **no quality-of-answer signal** for the GPU run — only that answers were returned, not whether they were correct or relevant.

### 11. PDF complex content is untested
The two PDFs tested are strictly text-heavy scientific papers. Embedded images inside PDFs, multi-column layouts, and complex tables are explicitly listed as not yet covered. We only have partial PDF confidence.

---

## 🔵 Architectural Fragility — Engineering Debt

### 12. ModelManager state corruption is fixed but not hardened
The reranker `try/except` fix prevents the crash, but the underlying design — where a failed model load corrupts the manager's state for the entire session — is still present. If any other model fails to load, the same class of bug is waiting to surface elsewhere.

### 13. No cross-modal retrieval has ever been attempted
A core RAG capability — answering a question that requires pulling one chunk from a PDF and one from an audio transcript simultaneously — has never been tested. This is explicitly listed as a success criterion and a major benchmark gap.

### 14. No robustness testing
Ambiguous queries, out-of-domain questions, and adversarial inputs have never been run against the pipeline. We do not know if the system hedges appropriately or simply hallucinates when context is absent.

---

## 📋 Summary Status Table

| Issue | Category | Fixed? |
|-------|----------|--------|
| **PaddleOCR `cls=True` bug** | Bug | ✅ |
| **Reranker NoneType crash** | Bug | ✅ |
| ONNX on CPU (embed + rerank) | Critical | ❌ |
| Image pipeline never re-run post-fix | Critical | ❌ |
| Audio retrieval untested | Critical | ❌ |
| DOCX/Markdown have zero coverage | Critical | ❌ |
| LLM still 5–7x above latency target | Performance | ❌ |
| KV cache VRAM offload unconfirmed | Performance | ❌ |
| No model pre-warming | Performance | ❌ |
| ONNX GPU not mathematically verified | Validation | ❌ |
| TTFT not measured | Validation | ❌ |
| No RAGAS metrics for GPU run | Validation | ❌ |
| PDF complex tables/images untested | Validation | ❌ |
| ModelManager fragility | Architecture | ❌ |
| Cross-modal retrieval never tested | Architecture | ❌ |
| No robustness/ambiguity testing | Architecture | ❌ |

> **Conclusion:** The two fixes completed so far were merely prerequisites to being able to run a proper benchmark — not the benchmark itself. The core system has never completed a full end-to-end validated run across all five modalities.
