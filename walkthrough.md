# Motif RAG — Phases 0–6 Audit & Real-Model Benchmarking Plan

---

## Part 1: System-Wide Audit (Phases 0–6)

### Test Suite Baseline

```
pytest tests/ -q --tb=short
→  207 passed, 34 skipped (models not downloaded), 0 failures
```

All 34 skips are `@pytest.mark.slow` tests that require models to be on disk.
Zero bugs or regressions. Full import audit also passes cleanly.

---

### Phase 0 — Infrastructure ✅

| Item | File | Status |
|---|---|---|
| `motif` CLI entry point | `rag/cli.py` | ✅ |
| REPL loop (prompt_toolkit) | `rag/cli.py` | ✅ |
| Config dataclasses + tier detection | `rag/config.py` | ✅ |
| Session save / load / trim | `rag/session.py` | ✅ |
| All slash commands routed | `rag/commands/` | ✅ |
| Model download script | `setup_models.py` | ✅ |
| Shared types (`Chunk`, `Citation`, etc.) | `rag/types.py` | ✅ |
| Query cache warning in welcome screen | `rag/cli.py` | ✅ |

### Phase 1 — Storage Layer ✅

| Item | File | Status |
|---|---|---|
| ChunkStore (SQLite, WAL mode) | `rag/storage/chunk_store.py` | ✅ |
| BM25Index (rank_bm25 + tantivy auto-switch) | `rag/retrieval/bm25_index.py` | ✅ |
| Ingestion Tracker (file hash dedup) | `rag/storage/ingestion_tracker.py` | ✅ |
| ModelManager (lazy-loading singletons) | `rag/models/model_manager.py` | ✅ |
| Embedder / Reranker / Captioner wrappers | `rag/models/` | ✅ |

### Phase 2 — Ingestion Pipeline ✅

| Item | File | Status |
|---|---|---|
| PDF, Markdown, Text, DOCX parsers | `rag/ingestion/parsers/` | ✅ |
| Image parser (PaddleOCR + captioning) | `rag/ingestion/parsers/image.py` | ✅ |
| Audio parser (whisper.cpp + timestamps) | `rag/ingestion/parsers/audio.py` | ✅ |
| OCR fallback in PDFParser | `rag/ingestion/parsers/pdf.py` | ✅ |
| Sentence chunker + semantic chunker | `rag/ingestion/` | ✅ |
| VectorStore (Qdrant in-proc) | `rag/retrieval/vector_store.py` | ✅ |
| Content-hash deduplication | `rag/ingestion/__init__.py` | ✅ |

### Phase 3 — Query Pipeline ✅

| Item | File | Status |
|---|---|---|
| QueryExpander (dense embed + HyDE) | `rag/retrieval/expander.py` | ✅ |
| Dense + BM25 retrieval | `rag/retrieval/` | ✅ |
| RRF fusion | `rag/retrieval/fusion.py` | ✅ |
| Cross-encoder reranker | `rag/reranking/cross_encoder.py` | ✅ |
| Context builder + citations | `rag/generation/` | ✅ |
| LLMClient (streaming, stop tokens) | `rag/generation/llm_client.py` | ✅ |
| QueryPipeline + filter modifiers | `rag/pipeline.py` | ✅ |

### Phase 4 — Quality & Hardening ✅

| Item | File | Status |
|---|---|---|
| HyDE query expansion | `rag/retrieval/expander.py` | ✅ |
| Semantic chunker | `rag/ingestion/semantic_chunker.py` | ✅ |
| QueryCache (SQLite LRU, enabled-flag guard) | `rag/storage/query_cache.py` | ✅ |
| Cache integrated in pipeline | `rag/pipeline.py` | ✅ |
| RAGAS dataset generator | `rag/evaluation/test_generator.py` | ✅ |

### Phase 5 — Multimodal ✅

| Item | File | Status |
|---|---|---|
| PaddleOCR image OCR (lazy import) | `rag/ingestion/parsers/image.py` | ✅ |
| Moondream2 captioning (T3 opt-in) | `rag/models/captioner.py` | ✅ |
| Whisper audio transcription + timestamps | `rag/ingestion/parsers/audio.py` | ✅ |
| Scanned PDF OCR fallback | `rag/ingestion/parsers/pdf.py` | ✅ |

### Phase 6 — Evaluation & Production Hardening ✅

| Item | File | Status |
|---|---|---|
| RAGAS offline runner (local LLM judge) | `rag/evaluation/ragas_runner.py` | ✅ |
| Latency test API (P50/P95/P99) | `rag/evaluation/latency_test.py` | ✅ |
| BM25 tantivy backend (≥ 100K auto-switch) | `rag/retrieval/bm25_index.py` | ✅ |
| test_cache.py (9 tests) | `tests/integration/` | ✅ |
| test_latency.py (5 + 1 slow) | `tests/integration/` | ✅ |

---

## Part 2: Real-Model Benchmarking Plan

This is the **standard industry procedure** for evaluating an offline RAG system.

---

### Step 0: Download Models

```powershell
motif setup

# Verify
python -c "
from rag.config import load_config; from pathlib import Path
cfg = load_config()
for name, path in [('LLM', cfg.models.llm_path), ('Embed', cfg.models.embed_model), ('Reranker', cfg.models.reranker)]:
    p = Path(path)
    print(f'{name}: {chr(10005) if p.exists() else \"MISSING\"} ({path})')
"
```

---

### Step 1: Ingest a Benchmark Corpus

```powershell
# Ingest your documents (recursive)
motif ingest ./your_docs/ -r

# Verify
motif status
# Expected: > 500 chunks indexed
```

> [!IMPORTANT]
> Use a **fixed corpus** for all benchmark runs. Changing the corpus invalidates
> the baseline. Recommended: 30–100 PDFs/MDs from a single clear domain.

---

### Step 2: Generate Synthetic Q&A Dataset (RAG-QA Generation)

```powershell
python -m rag.evaluation.test_generator -n 50 -o results/eval_dataset.json
```

This uses your local LLM to generate factual, answerable questions from random corpus chunks. Ground truth = source chunk text.

> [!TIP]
> Manually review 10–15 questions. Remove any that are ambiguous, leading, or
> unanswerable from the chunk alone. Quality here directly drives RAGAS reliability.

---

### Step 3: Run RAGAS Evaluation (Industry Standard)

RAGAS is the **de-facto standard** for RAG evaluation. It measures three orthogonal dimensions:

| Metric | What it asks | Target |
|---|---|---|
| **Faithfulness** | Every claim grounded in retrieved context? | ≥ 0.85 (T2/T3), ≥ 0.75 (T1) |
| **Answer Relevancy** | Does the answer address the question? | ≥ 0.80 |
| **Context Precision** | Are the retrieved passages relevant? | ≥ 0.75 |

```powershell
python -m rag.evaluation.ragas_runner -n 50 -o results/ragas_baseline.json

# View results
python -c "
import json
with open('results/ragas_baseline.json') as f: r = json.load(f)
print('=== RAGAS Baseline ===')
for k, v in r.items():
    print(f'  {k:25s}: {v:.3f}' if isinstance(v, float) else f'  {k:25s}: {v}')
"
```

---

### Step 4: Run Latency Benchmark

```powershell
python -c "
from rag.config import load_config
from rag.evaluation.latency_test import run_latency_test
import json

cfg = load_config()
with open('results/eval_dataset.json') as f:
    ds = json.load(f)

questions = [item['question'] for item in ds[:20]]
results = run_latency_test(questions, cfg, warmup=3)

print('=== Latency Results ===')
for k in ['tier', 'n_queries', 'p50_ms', 'p95_ms', 'p99_ms', 'max_ms']:
    print(f'  {k:12s}: {results[k]:.0f}' if isinstance(results[k], float) else f'  {k:12s}: {results[k]}')

targets = {'T1': 11000, 'T2': 8000, 'T3': 5000}
target = targets.get(results['tier'], 11000)
status = 'PASS' if results['p95_ms'] <= target else 'FAIL'
print(f'  NFR-07 P95 target ({target} ms): {status}')
"
```

---

### Step 5: Disk & RAM Footprint

```powershell
# Disk (NFR-01: total <= 5 GB)
python -c "
from pathlib import Path
def du(p): return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e9 if Path(p).exists() else 0
models = du('models')
index  = du(Path.home() / '.ragdb')
print(f'Models: {models:.2f} GB  |  Index: {index:.2f} GB  |  Total: {models+index:.2f} GB (target <= 5 GB)')
"
```

---

### Step 6: Cold-Start Time

```powershell
# NFR-03: cold start <= 3 seconds
Measure-Command { echo "exit" | motif } | Select-Object TotalSeconds
```

---

### Step 7: Slow Integration Tests (With Models)

```powershell
# Runs all @pytest.mark.slow tests — needs models downloaded
pytest tests/ -v -m slow --tb=short
```

---

### Step 8: Save Baseline & Regression Protocol

```powershell
$date = Get-Date -Format "yyyyMMdd"
Copy-Item results/ragas_baseline.json "results/ragas_baseline_$date.json"
```

For future changes, re-run Steps 3–4 and compare:

```python
# Any metric dropping > 2% = regression
for k in ['faithfulness','answer_relevancy','context_precision']:
    delta = new[k] - baseline[k]
    print(f'{k}: {\"REGRESSION\" if delta < -0.02 else \"OK\"}  ({delta:+.3f})')
```

---

## NFR Targets Summary

| NFR | Requirement | How to test |
|---|---|---|
| NFR-01 | Disk ≤ 5 GB | Step 5 |
| NFR-03 | Cold start ≤ 3 s | Step 6 |
| NFR-07 T1 | P95 ≤ 11 000 ms | Step 4 |
| NFR-07 T2 | P95 ≤ 8 000 ms | Step 4 |
| NFR-07 T3 | P95 ≤ 5 000 ms | Step 4 |
| RAG-F | Faithfulness ≥ 0.85 (T2/T3) | Step 3 RAGAS |
| RAG-R | Answer Relevancy ≥ 0.80 | Step 3 RAGAS |
| RAG-P | Context Precision ≥ 0.75 | Step 3 RAGAS |
| NFR-09 | 1000 queries no crash | `run_latency_test(questions * 50, cfg)` |
