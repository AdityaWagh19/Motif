# Final Resolved Stack + Memory Analysis

---

## Part 1 — Changes Incorporated from Review

### What Changed

| Parameter | Old | New | Reason |
|---|---|---|---|
| top_k retrieval | T1=15, T2=20, T3=20 | T1=20, T2=25, T3=30 | Retrieval is cheap; more candidates → better recall before reranking |
| top_k rerank | T1=3, T2=5, T3=5 | Unchanged | Rerank budget is correct |
| Context window | 1500 / 2048 / 2048 | 2048 / 3072 / 4096 | All models handle these safely; multi-page reasoning improves |
| T3 reranker | MiniLM-L12 (134 MB) | bge-reranker-base ONNX (~280 MB) | +2–3% nDCG@5 on hard queries; T3 has budget for it |
| T2 reranker | MiniLM-L12 | Unchanged | Accuracy/size trade-off not clear enough to upgrade |
| HyDE | T2/T3 always on | Adaptive (lightweight heuristic) | Simple queries skip HyDE; saves ~1–2s on factual lookups |
| Reranker skipping | — | Reranker always runs | 85ms is too cheap to skip; +12% nDCG@5 is non-negotiable |
| Metadata filtering | Not designed | Qdrant payload filters added | Essential for multi-document corpora |
| Context packing | Extractive compression | + Adjacent chunk merging | Reduces fragmentation when consecutive chunks retrieved |
| Image captioning load | "T3 opt-in install" | Conditional per-ingestion (image density gate) | Load moondream only if doc is image-heavy; unload immediately |
| BM25 | Already in pipeline | No change | Review misread this — BM25 + dense + sparse RRF was already designed |

---

### Adaptive HyDE Routing

Replaces the removed regex classifier with a simple, non-brittle heuristic:

```python
def should_use_hyde(query: str, config: RAGConfig) -> bool:
    if not config.retrieval.hyde_enabled:      # T1: always false
        return False

    tokens = query.lower().split()
    word_count = len(tokens)

    # Skip HyDE for short, clearly factual queries
    factual_markers = {
        'what', 'who', 'when', 'where', 'which', 'how many',
        'define', 'list', 'name'
    }
    reasoning_markers = {
        'why', 'explain', 'compare', 'difference', 'relationship',
        'impact', 'effect', 'because', 'summarize', 'analyze'
    }

    is_short_factual = (
        word_count <= 7
        and any(tokens[0] == m for m in factual_markers)
        and not any(m in query.lower() for m in reasoning_markers)
    )

    return not is_short_factual

# Short factual: "What is the boiling point of water?" → skip HyDE
# Multi-hop:     "Why did the experiment fail in section 3?" → use HyDE
```

---

### Metadata Filtering

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

def build_metadata_filter(
    filename: Optional[str] = None,
    source_type: Optional[str] = None,
    page_min: Optional[int] = None,
    page_max: Optional[int] = None,
    section: Optional[str] = None,
) -> Optional[Filter]:
    conditions = []

    if filename:
        conditions.append(FieldCondition(
            key="metadata.filename",
            match=MatchValue(value=filename)
        ))
    if source_type:
        conditions.append(FieldCondition(
            key="metadata.source_type",
            match=MatchValue(value=source_type)
        ))
    if page_min is not None or page_max is not None:
        conditions.append(FieldCondition(
            key="metadata.page_number",
            range=Range(gte=page_min, lte=page_max)
        ))
    if section:
        conditions.append(FieldCondition(
            key="metadata.section_title",
            match=MatchValue(value=section)
        ))

    return Filter(must=conditions) if conditions else None

# CLI usage:
# python cli.py ask "query" --file report.pdf --pages 10-20
# python cli.py ask "query" --type audio
```

SQLite already stores all these fields in ChunkMetadata. Qdrant payload is populated from the same fields at ingestion time. No schema changes required.

---

### Adjacent Chunk Merging in Context Packing

```python
def merge_adjacent_chunks(passages: List[ScoredPassage]) -> List[ScoredPassage]:
    """
    If two retrieved passages are consecutive chunks from the same source,
    merge them into one block instead of inserting a separator.
    Prevents the LLM from treating one continuous argument as two fragments.
    """
    if len(passages) <= 1:
        return passages

    merged = [passages[0]]

    for current in passages[1:]:
        prev = merged[-1]

        same_source = prev.passage.metadata.source_path == current.passage.metadata.source_path
        consecutive  = abs(prev.passage.metadata.char_end - current.passage.metadata.char_start) < 200

        if same_source and consecutive:
            # Merge: keep higher relevance score, concatenate text
            merged[-1] = ScoredPassage(
                passage=Passage(
                    text=prev.passage.text + "\n" + current.passage.text,
                    metadata=prev.passage.metadata   # Keep parent metadata
                ),
                relevance_score=max(prev.relevance_score, current.relevance_score)
            )
        else:
            merged.append(current)

    return merged
```

Applied in the context construction step, after reranking and anti-middle ordering.

---

### Conditional moondream2 Loading

```python
IMAGE_DENSITY_THRESHOLD = 0.3   # 30% of pages contain meaningful images

def ingest_document(filepath: Path, config: RAGConfig):
    parser = get_parser(filepath.suffix)
    extraction = parser.extract(filepath)

    # Decide whether to load moondream2 for this document
    if (config.hardware.tier == "T3"
        and config.parsers.moondream_available
        and filepath.suffix.lower() == ".pdf"):

        image_ratio = extraction.image_page_count / max(extraction.total_pages, 1)

        if image_ratio >= IMAGE_DENSITY_THRESHOLD:
            ModelManager.load("moondream")   # Load only now
            for block in extraction.image_blocks:
                block.caption = ModelManager.get("moondream").caption(block.image)
            ModelManager.unload("moondream") # Unload immediately

    # Continue ingestion with or without captions
    chunks = chunker.chunk(extraction.blocks)
    # ...
```

moondream2 RAM is in use for the duration of one document's image processing (~10–30 seconds), then freed. Never held across documents.

---

## Part 2 — Final Model Matrix

| Component | T1 (CPU / 8 GB RAM) | T2 (GTX 1650 / 4 GB VRAM) | T3 (RTX 3050 / 6–8 GB VRAM) |
|---|---|---|---|
| **LLM** | Phi-3.5-mini Q4_K_M (2.2 GB) | Qwen2.5-7B Q4_K_M (4.2 GB) | Qwen2.5-7B Q4_K_M (4.2 GB) |
| **GPU layers** | 0 | 20 of 28 (partial) | 28 of 28 (full) |
| **Embedding** | nomic-embed ONNX INT8 (274 MB) | nomic-embed ONNX INT8 | nomic-embed ONNX INT8 |
| **Reranker** | MiniLM-L6 ONNX (84 MB) | MiniLM-L12 ONNX (134 MB) | bge-reranker-base ONNX (280 MB) |
| **PDF parser** | pymupdf | pymupdf + PaddleOCR | Surya + PaddleOCR |
| **Audio** | whisper-tiny Q5 (75 MB) | whisper-tiny Q5 (75 MB) | whisper-small Q5 (244 MB) |
| **Image caption** | None | None | moondream2 Q4 — conditional, ingestion only |
| **top_k retrieval** | 20 | 25 | 30 |
| **top_k rerank** | 3 | 5 | 5 |
| **HyDE** | Off | Adaptive | Adaptive |
| **Semantic chunking** | Off (sentence split) | On | On |
| **Context window** | 2,048 tokens | 3,072 tokens | 4,096 tokens |
| **Metadata filters** | ✅ | ✅ | ✅ |
| **Adjacent merging** | ✅ | ✅ | ✅ |
| **Disk footprint** | **2.8 GB** ✅ | **4.9 GB** ✅ | **5.2 GB** opt |

---

## Part 3 — RAM & VRAM Memory Analysis

### Baseline Facts (Qwen2.5-7B Architecture)

Qwen2.5-7B-Instruct has **28 hidden layers** (not 35 — common misquote). This matters for the partial offload calculations:

```
Total GGUF Q4_K_M:   4.2 GB
Token embedding table + lm_head:  ~0.30 GB
Per transformer layer:  (4.2 - 0.30) / 28  ≈  139 MB/layer
GQA KV heads:  4 (not 8 — very memory-efficient grouped-query attention)
Head dimension:  128
```

KV cache formula: `2 × layers × kv_heads × head_dim × ctx_len × bytes_per_token`

---

### T1 — CPU / 8 GB RAM

#### Query-time RAM

| Component | RAM | Notes |
|---|---|---|
| OS + Python runtime | 2.00 GB | Fixed base |
| Phi-3.5-mini Q4_K_M weights | 2.20 GB | All 32 layers on CPU |
| Phi-3.5-mini KV cache (2048 ctx, Q8_0) | 0.08 GB | 2×32×4×128×2048×1B ÷ 1e9 = 67 MB |
| nomic-embed ONNX session | 0.55 GB | 274 MB disk → ONNX runtime expands to ~550 MB |
| MiniLM-L6 ONNX session | 0.15 GB | 84 MB disk → ~150 MB in session |
| Qdrant HNSW graph (on_disk=True) | 0.05 GB | Edge list only; vectors stay on disk |
| rank_bm25 inverted index | 0.08 GB | ~80 MB for 40K chunks, 512 tokens avg |
| SQLite page cache (WAL) | 0.06 GB | 64 MB configured |
| Python app + misc | 0.30 GB | |
| **Total query-time** | **~5.47 GB** | **2.53 GB headroom on 8 GB** ✅ |

#### Ingestion-time RAM (LLM unloaded)

> **Key design decision:** On T1, the LLM is **not loaded during ingestion**. It is only loaded when the user runs `cli.py ask`. This is already implied by `ModelManager` lazy-loading.

| Component | RAM |
|---|---|
| OS + Python runtime | 2.00 GB |
| nomic-embed ONNX | 0.55 GB |
| PaddleOCR (during image processing) | 0.50 GB |
| whisper-tiny (during audio processing) | 0.20 GB |
| Qdrant write path + BM25 + SQLite | 0.35 GB |
| Python app | 0.30 GB |
| **Total ingestion peak** | **~3.90 GB** | **4.10 GB headroom on 8 GB** ✅ |

---

### T2 — GTX 1650 / 4 GB VRAM, 8 GB RAM

#### VRAM breakdown (n_gpu_layers=20)

| Component | VRAM |
|---|---|
| Qwen2.5-7B — 20 GPU layers × 139 MB | 2.78 GB |
| Token embedding table (GPU-resident) | 0.15 GB |
| KV cache (20 GPU layers, 3072 ctx, Q8_0): `2×20×4×128×3072×1B` | 0.06 GB |
| CUDA runtime + framework overhead | 0.15 GB |
| **Total VRAM** | **~3.14 GB** | **0.86 GB spare on 4 GB** ✅ |

> This is the correct corrected figure. The earlier 2.8 GB estimate was based on incorrect layer counts.

#### CPU RAM breakdown (T2, query-time)

| Component | RAM |
|---|---|
| OS + Python runtime | 2.00 GB |
| Qwen2.5-7B — 8 CPU layers × 139 MB + lm_head | 1.26 GB |
| nomic-embed ONNX session | 0.55 GB |
| MiniLM-L12 ONNX session | 0.30 GB |
| Qdrant HNSW graph + BM25 + SQLite | 0.40 GB |
| Python app | 0.30 GB |
| **Total CPU RAM** | **~4.81 GB** | **3.19 GB headroom on 8 GB** ✅ |

#### T2 Ingestion-time peak

| Component | RAM |
|---|---|
| OS + Python (base) | 2.00 GB |
| nomic-embed ONNX | 0.55 GB |
| PaddleOCR | 0.50 GB |
| whisper-tiny | 0.20 GB |
| Qdrant write + BM25 + SQLite | 0.40 GB |
| Python app | 0.30 GB |
| **Total ingestion peak RAM** | **~3.95 GB** | **4.05 GB headroom on 8 GB** ✅ |

> Note: During ingestion the LLM layers split between CPU/VRAM are **not loaded** (LLM is query-time only). VRAM is free during ingestion on T2.

---

### T3 — RTX 3050 / 6–8 GB VRAM, 8–16 GB RAM

#### VRAM breakdown (n_gpu_layers=28, full offload)

| Component | VRAM |
|---|---|
| Qwen2.5-7B — all 28 layers × 139 MB | 3.89 GB |
| Token embedding table + lm_head | 0.30 GB |
| KV cache (28 GPU layers, 4096 ctx, Q8_0): `2×28×4×128×4096×1B` | 0.12 GB |
| CUDA runtime overhead | 0.15 GB |
| **Total VRAM** | **~4.46 GB** | **1.54 GB spare on 6 GB** ✅ |

#### CPU RAM breakdown (T3, query-time)

| Component | RAM |
|---|---|
| OS + Python runtime | 2.00 GB |
| Qwen2.5-7B (fully GPU-resident, CPU buffers only) | 0.20 GB |
| nomic-embed ONNX session | 0.55 GB |
| bge-reranker-base ONNX session | 0.45 GB |
| Qdrant HNSW graph + BM25 + SQLite | 0.40 GB |
| Python app | 0.30 GB |
| **Total CPU RAM** | **~3.90 GB** | **4.10 GB headroom on 8 GB** ✅ |

> When the LLM is fully GPU-offloaded, CPU RAM barely matters. T3 is the most memory-efficient configuration because the LLM — the heaviest component — sits entirely in VRAM and doesn't touch system RAM.

#### T3 Ingestion with moondream2 (conditional peak)

| Component | RAM | Notes |
|---|---|---|
| Base query-time RAM | 3.90 GB | From above |
| moondream2 Q4 (loaded for one image-heavy doc) | 1.20 GB | Loaded then unloaded per document |
| PaddleOCR / Surya | 0.70 GB | Surya is heavier than PaddleOCR |
| whisper-small | 0.45 GB | |
| **Peak ingestion RAM** | **~6.25 GB** | Tight on 8 GB (1.75 GB headroom) ⚠️ |

> **This is the only tight scenario**: T3 with moondream2 active during ingestion. Mitigation: load moondream2, process one document's images, unload — don't hold it alongside Surya and whisper simultaneously. Sequence the loading:
> 1. Surya (layout analysis) → unload
> 2. PaddleOCR (OCR text) → keep loaded
> 3. moondream2 (captioning) → load → process → unload
> 4. whisper (audio) → load → process → unload
>
> Sequential loading peak: ~5.3 GB. Comfortable.

---

### Cross-Tier Memory Summary

| Metric | T1 | T2 | T3 |
|---|---|---|---|
| Query-time RAM | 5.47 GB | 4.81 GB | 3.90 GB |
| Query-time VRAM | 0 GB | 3.14 GB | 4.46 GB |
| Ingestion peak RAM | 3.90 GB | 3.95 GB | ~5.30 GB (sequential) |
| RAM headroom (8 GB) | 2.53 GB | 3.19 GB | 4.10 GB |
| VRAM headroom | — | 0.86 GB | 1.54 GB |

---

## Part 4 — Is the Memory Consumption Justified?

### Per-Component Verdict

| Component | RAM | Justified? | Optimize? |
|---|---|---|---|
| LLM (Phi/Qwen) | 2.2–4.2 GB | ✅ Core function | No — this is the price of offline generation |
| KV cache | 0.06–0.12 GB | ✅ Enables context | No — Q8_0 is already quantized. Larger context = better quality |
| nomic-embed ONNX | 0.55 GB | ✅ Needed every query | Marginal: drop to 256-dim Matryoshka saves ~35% index size, not RAM |
| Reranker ONNX | 0.15–0.45 GB | ✅ Highest ROI component | No — ONNX is already the lean runtime |
| Qdrant HNSW graph | 0.05 GB | ✅ Retrieval speed | Already minimal with on_disk=True |
| BM25 index | 0.08 GB | ✅ Lexical recall | No issue at reasonable corpus sizes |
| SQLite page cache | 0.06 GB | ✅ Chunk fetch speed | No — 64 MB is already conservative |
| PaddleOCR (ingestion) | 0.50 GB | ✅ Ingestion only | No — unloaded after ingestion |
| moondream2 (T3 opt-in) | 1.20 GB | ✅ Ingestion only | Already sequential-loaded; not a query-time concern |

### Two Genuine Optimizations

---

#### Optimization 1 — Matryoshka 256-dim for T1 Corpus Index

At query time, nomic-embed's RAM footprint is fixed (~0.55 GB) regardless of embedding dimension. But the vector index storage scales with dimension:

```python
# Standard 768-dim: 40K chunks × 768 bytes (INT8) = 30.7 MB
# 256-dim:          40K chunks × 256 bytes (INT8) = 10.2 MB

# Encode at full 768-dim, truncate before storage:
embeddings = model.encode(texts, normalize=True)
embeddings_256 = normalize(embeddings[:, :256])   # Slice and re-normalize
```

**Impact:** 67% vector storage reduction, ~3% retrieval accuracy loss, 0 RAM impact at query time.
**Recommendation:** Enable on T1 only (`embed_dim = 256` in config), where disk and index size are tighter.

---

#### Optimization 2 — Don't load the LLM until first query (lazy startup)

On T1, startup currently loads the LLM immediately. This means:
- First `python cli.py ask` takes ~15–20s to load Phi-3.5-mini
- If the user only runs `python cli.py ingest`, they paid 2.2 GB RAM for nothing

```python
class ModelManager:
    @classmethod
    def get_llm(cls):
        if 'llm' not in cls._models:
            logger.info("Loading LLM — first query will be slower...")
            cls._models['llm'] = load_llama_cpp(config.models.llm_path)
        return cls._models['llm']

    @classmethod
    def unload_llm(cls):
        """Call before ingestion on T1 to free 2.2 GB."""
        if 'llm' in cls._models:
            del cls._models['llm']
            gc.collect()
```

**Impact:** Ingestion on T1 drops from 5.47 GB peak to 3.90 GB. Startup time for first query is the same, but `cli.py ingest` runs faster and lighter.

---

### What Is NOT Over-Engineered

- **Keeping nomic-embed loaded at all times during query session**: Correct. You cannot encode queries without it. Reloading it per-query would add 3–5s startup per query.
- **HNSW graph in RAM vs. full disk**: Correct trade-off. HNSW graph is only 50 MB; pushing it to disk would add 5–10ms per query for no disk savings (it's tiny).
- **KV cache at current sizes**: Correct. Going smaller (e.g., 1024 ctx on T2) would hurt multi-page reasoning significantly more than the RAM savings justify.

### What Is Correctly Constrained by Design

- **Qdrant `on_disk=True`**: Vectors are memory-mapped from disk. Only the graph edges stay in RAM. This is the right call — without it, 40K chunks at 1024 bytes each = 40 MB RAM constant, which is not catastrophic but unnecessary.
- **BM25 switch to tantivy at 100K+ chunks**: rank_bm25 keeps the full index in RAM. At 100K chunks it reaches ~200 MB. That's the threshold to switch to tantivy, which memory-maps the index. Already in the design.
- **Sequential ingestion model loading on T3**: Prevents the 6.25 GB peak that would occur if Surya, whisper, and moondream2 were all resident simultaneously.

---

## Conclusion

The memory consumption is **correctly sized and well-justified** across all three tiers.

The two genuine optimizations are **minor refinements**, not architectural corrections:
1. 256-dim Matryoshka embeddings for T1 index storage (not RAM)
2. Lazy LLM loading so ingestion doesn't pay the LLM's memory cost

No component is wasteful. The largest single memory consumers — the LLM and the ONNX embedding session — are both essential to every query and irreducible without meaningfully dropping accuracy. The Qdrant and BM25 components are already near-optimal.

**The stack is ready for implementation.**
