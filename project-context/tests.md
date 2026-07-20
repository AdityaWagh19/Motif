# Test Strategy & Evaluation Plan — Motif Offline Multimodal RAG

> **Depends on:** `trd.md`, `architecture.md`  
> **Purpose:** Defines unit tests, integration tests, evaluation methodology, and the regression protocol before any model or config change.

---

## 1. Unit Tests

All unit tests live in `tests/unit/`. Run with:
```powershell
pytest tests/unit/ -v
```

### 1.1 Chunker (`tests/unit/test_chunker.py`)

| Test | What It Checks | TRD ID |
|---|---|---|
| `test_sentence_chunker_size` | 95% of output chunks have token_count in [64, 640] | ING-10 |
| `test_sentence_chunker_no_mid_sentence` | No chunk ends mid-sentence (ends with `.`, `?`, `!`, `\n`) | ING-11 |
| `test_overlap_tokens` | Consecutive chunks share 40–80 overlapping tokens | ING-12 |
| `test_table_kept_intact` | A chunk containing `\|...\|` is never split | ING-13 |
| `test_semantic_chunker_size` | Same as sentence but with semantic boundary detection | ING-10 |
| `test_empty_input` | Empty string returns empty list, no crash | — |
| `test_single_short_block` | Block shorter than MIN_CHUNK_TOKENS returns one chunk | — |

### 1.2 Embedder (`tests/unit/test_embedder.py`)

| Test | What It Checks |
|---|---|
| `test_encode_returns_correct_shape` | Output shape is `(N, embed_dim)` with `embed_dim` = 768 (T2/T3) or 256 (T1) |
| `test_encode_normalized` | L2 norm of each vector is 1.0 ± 1e-5 |
| `test_encode_batch_matches_single` | Batch encoding matches single encoding per item |
| `test_matryoshka_truncation` | 256-dim truncated vector re-normalizes to norm 1.0 |
| `test_encode_empty_string` | Empty string encodes without crash; returns zero vector or raises ValueError |

### 1.3 BM25 Index (`tests/unit/test_bm25.py`)

| Test | What It Checks | TRD ID |
|---|---|---|
| `test_exact_phrase_in_top3` | A 3-word exact phrase from corpus ranks in BM25 top-3 | RET-07 |
| `test_add_and_search` | Add a chunk, search for a term in it, get that chunk back | — |
| `test_delete_removes_chunk` | After delete, chunk ID no longer appears in results | ING-19 |
| `test_rebuild_after_delete` | rank_bm25 index correctly rebuilt after deletion | — |

### 1.4 RRF Fusion (`tests/unit/test_fusion.py`)

| Test | What It Checks | TRD ID |
|---|---|---|
| `test_rrf_output_count` | Returns exactly `top_k` results | RET-04 |
| `test_multi_list_boost` | Chunk in all 3 lists ranks higher than chunk in 1 list | RET-05 |
| `test_rrf_k_parameter` | Changing k=60 to k=30 changes score magnitudes but not ranking direction | — |
| `test_empty_list_handling` | If one list is empty, RRF still returns results from others | — |

### 1.5 Cross-Encoder Reranker (`tests/unit/test_reranker.py`)

| Test | What It Checks | TRD ID |
|---|---|---|
| `test_reranker_output_count` | Returns exactly `top_k` passages | — |
| `test_reranker_scores_in_range` | All relevance scores are in [0.0, 1.0] | — |
| `test_threshold_filters` | Passages below 0.3 threshold are excluded | RER-03 |
| `test_relevant_ranks_higher` | Semantically relevant passage scores higher than irrelevant | RER-02 |

### 1.6 Context Builder (`tests/unit/test_context_builder.py`)

| Test | What It Checks | TRD ID |
|---|---|---|
| `test_anti_middle_ordering` | Rank-1 passage is at index 0 in output string | GEN-07 |
| `test_adjacent_merge` | Two consecutive passages from same source become one block | GEN-08 |
| `test_token_budget_respected` | Output never exceeds `context_max_tokens` | GEN-06 |
| `test_top1_preserved_after_compress` | Extractive compression never drops the top-1 passage | GEN-09 |
| `test_extractive_compress_triggers` | Compression activates when raw context > budget | GEN-09 |

### 1.7 HyDE Routing (`tests/unit/test_expander.py`)

| Test | What It Checks |
|---|---|
| `test_short_factual_skips_hyde` | "What is X?" (5 words) → `should_use_hyde()` returns False |
| `test_reasoning_uses_hyde` | "Why did X happen?" → `should_use_hyde()` returns True |
| `test_t1_always_no_hyde` | On T1 config, always returns False regardless of query |
| `test_multi_query_expand` | `expand(mode='multi_query')` returns 3 distinct queries |

### 1.8 Citation Formatter (`tests/unit/test_citations.py`)

| Test | What It Checks | TRD ID |
|---|---|---|
| `test_pdf_citation_format` | PDF citation includes filename, page number, section | — |
| `test_audio_citation_format` | Audio citation includes filename + `@ MM:SS–MM:SS` | ING-08 |
| `test_md_citation_format` | Markdown citation includes filename + section | — |
| `test_image_citation_format` | Image citation includes filename (no page) | — |

### 1.9 Ingestion Tracker (`tests/unit/test_tracker.py`)

| Test | What It Checks | TRD ID |
|---|---|---|
| `test_not_indexed_initially` | Returns False for a file that hasn't been ingested | ING-17 |
| `test_indexed_after_update` | Returns True after `tracker.update(filepath, hash)` | ING-17 |
| `test_hash_change_detected` | Returns False after content hash changes | ING-18 |
| `test_remove_clears_entry` | Returns False after `tracker.remove(filepath)` | ING-20 |

### 1.10 Session History (`tests/unit/test_session.py`)

| Test | What It Checks | TRD ID |
|---|---|---|
| `test_add_turn_grows_history` | After `add_turn(q, a)`, `len(history) == 2` (one user + one assistant entry) | HST-01 |
| `test_rolling_window_trims_oldest` | With budget=512 tokens, oldest turns are dropped when limit exceeded | HST-02 |
| `test_passages_kept_over_history` | When budget forces a choice, retrieved passage is included and oldest history dropped | HST-03 |
| `test_save_creates_json` | After `session.save()`, `~/.ragdb/history.json` exists and is valid JSON | HST-04 |
| `test_load_restores_history` | After `session.load()`, `session.history` matches what was saved | HST-05 |
| `test_empty_history_no_error` | `Session()` with no history.json starts with `history == []` and no exception | HST-06 |
| `test_clear_resets_and_deletes` | After `session.clear()`, `history == []` and `history.json` does not exist | HST-07 |
| `test_new_archives_history` | After `session.new()`, `history_TIMESTAMP.json` exists and `history.json` is gone | CMD-07 |

---

## 2. Integration Tests

All integration tests live in `tests/integration/`. They require models to be downloaded. Run with:
```powershell
pytest tests/integration/ -v -m "not slow"    # Skip slow LLM tests
pytest tests/integration/ -v                   # Run all including LLM tests
```

### 2.1 End-to-End Ingestion (`tests/integration/test_ingestion.py`)

```python
# Test: ingest a small corpus and verify all indices populated
def test_full_ingestion_pipeline(tmp_path, sample_pdf, sample_markdown):
    config = load_config("T2")
    ingest_document(sample_pdf, config)
    ingest_document(sample_markdown, config)

    chunk_count = ChunkStore(config).count()
    qdrant_count = VectorStore(config).count()
    assert chunk_count > 0
    assert chunk_count == qdrant_count          # ING-15
    assert BM25Index(config).count() == chunk_count  # ING-16
    assert IngestionTracker(config).is_indexed(sample_pdf)  # ING-17
```

```python
# Test: re-ingestion adds zero chunks
def test_deduplication(tmp_path, sample_pdf):
    config = load_config("T2")
    ingest_document(sample_pdf, config)
    count_before = ChunkStore(config).count()

    ingest_document(sample_pdf, config)   # Second ingest
    count_after = ChunkStore(config).count()

    assert count_before == count_after    # ING-14, ING-18
```

### 2.2 End-to-End Query (`tests/integration/test_query.py`)

```python
# Test: answerable question returns a grounded answer
def test_answerable_query_returns_citation(seeded_corpus, config):
    pipeline = QueryPipeline(config)
    answer = pipeline.answer("What is the main topic of the document?")

    assert len(answer.text) > 20
    assert len(answer.citations) >= 1            # GEN-05
    assert answer.confidence > 0.3
```

```python
# Test: unanswerable question returns refusal, not hallucination
def test_unanswerable_returns_refusal(seeded_corpus, config):
    pipeline = QueryPipeline(config)
    answer = pipeline.answer("What is the boiling point of tungsten carbide?")

    refusal_phrases = ["not found", "not mentioned", "no information", "cannot find"]
    assert any(phrase in answer.text.lower() for phrase in refusal_phrases)  # GEN-04
```

### 2.3 Delete and Sync (`tests/integration/test_sync.py`)

```python
def test_delete_removes_all_chunks(seeded_corpus, filepath, config):
    count_before = VectorStore(config).count()
    chunk_count_for_file = ChunkStore(config).count_by_source(filepath)

    delete_document(filepath, config)

    assert VectorStore(config).count() == count_before - chunk_count_for_file  # ING-19
    assert not IngestionTracker(config).is_indexed(filepath)
```

---

## 3. Evaluation — RAGAS Metrics

### 3.1 Setting Up the Eval Dataset

**Before real corpus (proxy benchmark):**
```powershell
# Download 50 documents from FRAMES benchmark
python -m rag.evaluation.download_frames --output tests/eval_corpus/frames/

# Ingest them
motif ingest tests/eval_corpus/frames/ -r

# Run against FRAMES QA pairs
python -m rag.evaluation.ragas_runner \
  --dataset tests/eval_corpus/frames_qa.json \
  --output results/frames_baseline.json
```

**After real corpus is ingested:**
```powershell
# Generate synthetic QA pairs from your corpus
python -m rag.evaluation.test_generator \
  --n-questions 150 \
  --output tests/eval_corpus/synthetic_qa.json

# Run RAGAS evaluation
python -m rag.evaluation.ragas_runner \
  --dataset tests/eval_corpus/synthetic_qa.json \
  --output results/ragas_$(Get-Date -Format 'yyyyMMdd').json
```

### 3.2 RAGAS Metric Targets

| Metric | T1 Target | T2/T3 Target | Method |
|---|---|---|---|
| Faithfulness | ≥ 75% | ≥ 85% | RAGAS: does context entail the answer? |
| Answer Relevancy | ≥ 78% | ≥ 85% | RAGAS: does answer address the question? |
| Context Precision | ≥ 80% | ≥ 88% | RAGAS: are retrieved chunks relevant? |
| Context Recall | ≥ 75% | ≥ 80% | RAGAS: are all relevant facts retrieved? |
| Retrieval Recall@20 | ≥ 70% | ≥ 75% | Ground truth chunk in top-20 |

All RAGAS metrics use the local LLM as judge (`judge_model = "same"` in config).

### 3.3 Reading RAGAS Results

```json
{
  "faithfulness": 0.87,
  "answer_relevancy": 0.89,
  "context_precision": 0.91,
  "context_recall": 0.83,
  "n_samples": 150,
  "tier": "T2",
  "timestamp": "2025-07-18T08:00:00"
}
```

If `faithfulness < 0.75`:
1. Enable `--verbose` on queries to inspect retrieved passages
2. Check if passages actually contain the answer (retrieval miss vs. LLM error)
3. If retrieval miss: lower `relevance_threshold` to 0.2, increase `top_k_retrieval`
4. If LLM ignoring context: strengthen "answer only from context" in system prompt

---

## 4. Latency Benchmark

```powershell
python -m rag.evaluation.latency_test \
  --n-queries 100 \
  --dataset tests/eval_corpus/synthetic_qa.json \
  --output results/latency_$(Get-Date -Format 'yyyyMMdd').json
```

Expected output:
```
Tier: T2
HyDE: adaptive
────────────────────────────────
Stage              P50     P95
────────────────────────────────
Query encoding     18ms    24ms
Retrieval          45ms    82ms
Reranking          75ms   145ms
LLM (first token) 980ms  1420ms
LLM (full answer) 3200ms 4800ms
────────────────────────────────
End-to-end        4100ms 7200ms
```

**TRD Acceptance:**
- T1 P95 ≤ 13s (GEN-14)
- T2 P95 ≤ 8s (GEN-15)
- T3 P95 ≤ 5s (GEN-16)

---

## 5. Footprint Measurement

```powershell
# Measure model weight disk size
python -c "
import os
total = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, filenames in os.walk('models')
    for f in filenames
)
print(f'Model disk: {total/1e9:.2f} GB')
"

# Measure index disk size
python -c "
import os
import pathlib
db_path = pathlib.Path.home() / '.ragdb'
total = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
print(f'Index disk: {total/1e9:.2f} GB')
"
```

---

## 6. Regression Protocol

**Run before merging any change to:**
- Model selection or quantization
- Chunking strategy or parameters
- Retrieval parameters (top_k, threshold)
- Prompt templates
- Context construction logic

**Regression test suite:**
```powershell
# 1. Unit tests (fast, < 2 minutes)
pytest tests/unit/ -v

# 2. Integration tests (medium, < 10 minutes, requires models)
pytest tests/integration/ -v -m "not slow"

# 3. RAGAS snapshot comparison (slow, ~30 minutes)
python -m rag.evaluation.ragas_runner \
  --dataset tests/eval_corpus/synthetic_qa.json \
  --compare results/ragas_baseline.json \
  --output results/ragas_regression.json
# Fails if any metric drops > 2% from baseline
```

**Baseline snapshot:** After Phase 2 is complete and targets are met, run RAGAS and save the output as `results/ragas_baseline.json`. All future regression runs compare against this.

---

## 7. A/B Testing Framework

Used to compare pipeline configurations on the same eval dataset (Phase 2+):

```powershell
python -m rag.evaluation.ab_test \
  --config-a config_hyde_on.toml \
  --config-b config_hyde_off.toml \
  --dataset tests/eval_corpus/synthetic_qa.json \
  --output results/ab_hyde_$(Get-Date -Format 'yyyyMMdd').json
```

Output format:
```json
{
  "config_a": {"name": "HyDE on", "faithfulness": 0.87, "p95_latency_ms": 7200},
  "config_b": {"name": "HyDE off", "faithfulness": 0.83, "p95_latency_ms": 4800},
  "winner": "config_a",
  "faithfulness_delta": "+4.8%",
  "latency_delta": "+2400ms"
}
```

Primary A/B test targets (Phase 2):
1. HyDE vs. no HyDE on real corpus
2. MiniLM-L12 vs. bge-reranker-base (T3)
3. semantic chunking vs. sentence chunking (T2)
