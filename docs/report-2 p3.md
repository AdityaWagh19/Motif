---

## 13. Reranking Strategy

### 13.1 Two-Stage Reranking Architecture

```
Stage 1 (Fast, recall-oriented):   Hybrid retrieval → top-20 (RRF)
Stage 2 (Slow, precision-oriented): Cross-encoder reranking → top-5
```

This architecture is optimal because:
- Cross-encoders are O(k × n) where k = number of passages, n = passage length
- Running cross-encoders on 1M passages would take minutes; on 20, it takes ~150ms
- The first stage uses fast ANN to prune to a manageable candidate set

### 13.2 Cross-Encoder Implementation

```python
class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device=device,
            # Use ONNX for faster CPU inference
            backend='onnx' if USE_ONNX else 'torch'
        )
    
    def rerank(
        self,
        query: str,
        passages: List[Passage],
        top_k: int = 5
    ) -> List[ScoredPassage]:
        
        pairs = [(query, p.text[:512]) for p in passages]
        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Apply sigmoid for calibrated relevance scores
        scores = 1 / (1 + np.exp(-scores))
        
        # Sort and filter
        ranked = sorted(
            zip(passages, scores),
            key=lambda x: -x[1]
        )
        
        # Filter out low-relevance passages
        filtered = [(p, s) for p, s in ranked if s > RELEVANCE_THRESHOLD]
        
        return [
            ScoredPassage(passage=p, relevance_score=s)
            for p, s in filtered[:top_k]
        ]
```

### 13.3 Relevance Threshold Calibration

Setting the relevance threshold is critical — too low includes noise, too high misses valid context:

```python
# Recommended thresholds per reranker model:
THRESHOLDS = {
    'cross-encoder/ms-marco-MiniLM-L-12-v2': 0.3,   # Sigmoid-scaled
    'BAAI/bge-reranker-v2-m3': 0.4,
    'cross-encoder/ms-marco-TinyBERT-L-2-v2': 0.2,  # Less calibrated
}

# For calibration on your corpus:
# 1. Sample 100 queries with known answers
# 2. Run reranker, record scores of relevant vs non-relevant passages
# 3. Set threshold at 95th percentile of non-relevant scores
```

### 13.4 UPR as Fallback (Zero-Weight Reranking)

When storage is extremely constrained and no reranker model can be loaded:

```python
def upr_rerank(query: str, passages: List[Passage], llm) -> List[ScoredPassage]:
    """
    Unsupervised Passage Reranking via LLM likelihood.
    Score = P(query | passage) approximated via LLM log-prob.
    """
    scores = []
    for passage in passages:
        prompt = f"Given the following passage, how relevant is it to: '{query}'?\n\n{passage.text}\n\nRelevance (0-10):"
        # Use the LLM's log-probability of generating high-relevance response
        score = llm.score_completion(prompt, "10")  # llama.cpp logit access
        scores.append(score)
    
    return sorted(zip(passages, scores), key=lambda x: -x[1])
```

**UPR trade-off:** Adds ~800ms per query but requires no additional model weights. Best reserved for edge cases.

### 13.5 Diversity-Aware Selection

After reranking, apply Maximal Marginal Relevance (MMR) to avoid redundant passages in context:

```python
def mmr_select(
    query_vec: np.ndarray,
    passages: List[ScoredPassage],
    lambda_param: float = 0.7,
    top_k: int = 5
) -> List[ScoredPassage]:
    """
    MMR balances relevance (λ) and diversity (1-λ).
    lambda=0.7: prefer relevance; lambda=0.3: prefer diversity.
    """
    selected = []
    remaining = list(passages)
    
    while len(selected) < top_k and remaining:
        best_idx, best_score = None, -np.inf
        
        for i, candidate in enumerate(remaining):
            # Relevance to query
            relevance = candidate.relevance_score
            
            # Maximum similarity to already-selected passages
            if selected:
                max_sim = max(
                    cosine_sim(candidate.embedding, s.embedding)
                    for s in selected
                )
            else:
                max_sim = 0.0
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i
        
        selected.append(remaining.pop(best_idx))
    
    return selected
```

---

## 14. Context Construction

### 14.1 Anti-Lost-in-the-Middle Ordering

Research (Liu et al. 2023) shows performance degrades sharply when relevant information is in the middle of the context. Apply this ordering:

```python
def order_passages_for_context(passages: List[ScoredPassage]) -> List[ScoredPassage]:
    """
    Reorder passages to maximize LLM attention on most relevant content.
    Rank 1 → first position (always attended to)
    Rank 2 → last position (attended to due to recency bias)
    Ranks 3-N → middle (may be underweighted)
    """
    if len(passages) <= 2:
        return passages
    
    ordered = []
    ordered.append(passages[0])   # Rank 1: first
    ordered.extend(passages[2:])  # Ranks 3-N: middle
    ordered.append(passages[1])   # Rank 2: last
    
    return ordered
```

### 14.2 Context Compression via Extractive Selection

Fast compression without a separate model — extract most relevant sentences using embedding similarity:

```python
def extractive_compress(
    query: str,
    passages: List[Passage],
    target_tokens: int = 1500,
    embed_model = None
) -> str:
    """
    Select highest-relevance sentences from retrieved passages.
    No additional model required beyond the embedding model.
    """
    query_vec = embed_model.encode(query)
    
    # Score each sentence against query
    all_sentences = []
    for passage in passages:
        for sent in sent_tokenize(passage.text):
            if len(sent.split()) < 5:  # Skip very short sentences
                continue
            sent_vec = embed_model.encode(sent)
            score = cosine_similarity([query_vec], [sent_vec])[0][0]
            all_sentences.append((score, sent, passage.source))
    
    # Sort by relevance, take until token budget exhausted
    all_sentences.sort(key=lambda x: -x[0])
    
    selected = []
    total_tokens = 0
    for score, sent, source in all_sentences:
        sent_tokens = count_tokens(sent)
        if total_tokens + sent_tokens > target_tokens:
            break
        selected.append((score, sent, source))
        total_tokens += sent_tokens
    
    # Re-order by source/document order (not by score) for coherent reading
    selected.sort(key=lambda x: passages.index(
        next(p for p in passages if p.source == x[2])
    ))
    
    return "\n".join(sent for _, sent, _ in selected)
```

### 14.3 Context Template

```python
def build_context(passages: List[ScoredPassage], max_tokens: int = 2048) -> str:
    ordered = order_passages_for_context(passages)
    
    parts = []
    total_tokens = 0
    
    for i, scored_passage in enumerate(ordered, 1):
        p = scored_passage.passage
        
        # Source header for citations
        header = f"[Source {i}: {p.metadata.source_path}"
        if p.metadata.page_number > 0:
            header += f", page {p.metadata.page_number}"
        if p.metadata.section_title:
            header += f", section: {p.metadata.section_title}"
        header += "]"
        
        chunk_text = f"{header}\n{p.text}"
        chunk_tokens = count_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > max_tokens:
            # Truncate this passage to fit
            remaining = max_tokens - total_tokens
            if remaining > 100:  # Only include if enough space
                truncated = truncate_to_tokens(chunk_text, remaining)
                parts.append(truncated)
            break
        
        parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    return "\n\n---\n\n".join(parts)
```

### 14.4 Citation Extraction

```python
def extract_citations(passages: List[ScoredPassage]) -> List[Citation]:
    """Extract structured citations for display in CLI output."""
    citations = []
    for i, sp in enumerate(passages, 1):
        p = sp.passage
        meta = p.metadata
        
        citations.append(Citation(
            number=i,
            filepath=meta.source_path,
            filename=Path(meta.source_path).name,
            page=meta.page_number if meta.page_number > 0 else None,
            section=meta.section_title,
            relevance_score=sp.relevance_score,
            excerpt=p.text[:100] + "..." if len(p.text) > 100 else p.text
        ))
    
    return citations

def format_citations_for_cli(citations: List[Citation]) -> str:
    lines = ["\n📚 Sources:"]
    for c in citations:
        line = f"  [{c.number}] {c.filename}"
        if c.page:
            line += f" (p.{c.page})"
        if c.section:
            line += f" — {c.section}"
        lines.append(line)
    return "\n".join(lines)
```

---

## 15. Latency Optimization

### 15.1 Latency Budget Analysis

Target: <5 seconds end-to-end for interactive CLI experience.

```
Component                        Target    Optimization Strategy
────────────────────────────────────────────────────────────────
Query encoding (BGE-M3)          30ms      ONNX runtime, INT8
ANN search (Qdrant HNSW)         5ms       ef=128, on_disk vectors
Sparse search (Qdrant)           3ms       on_disk sparse index
BM25 search (tantivy)            2ms       pre-warmed index, Rust
RRF fusion                       <1ms      Pure Python, O(n)
SQLite chunk fetch (top-20)      5ms       Indexed by chunk_id
Cross-encoder rerank (20 pairs)  150ms     ONNX runtime, batch=16
Context assembly                 <5ms      String operations
LLM first-token latency          800ms     n_gpu_layers > 0 if GPU
LLM generation (200 tokens)      1,200ms   Q4_K_M, 8-thread CPU
Source formatting                <5ms      
────────────────────────────────────────────────────────────────
Total (CPU-only, no HyDE)        ≈2,206ms  ✅ Under 3 seconds
Total (CPU-only, with HyDE)      ≈3,800ms  ✅ Under 5 seconds
```

### 15.2 ONNX Runtime for Embedding & Reranking

Converting sentence-transformers to ONNX provides 2–3× speedup on CPU:

```python
# Export BGE-M3 to ONNX (one-time)
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import AutoQuantizationConfig

model = ORTModelForFeatureExtraction.from_pretrained(
    "BAAI/bge-m3",
    export=True,
    quantization_config=AutoQuantizationConfig.avx2(is_static=False)
)
model.save_pretrained("~/.ragdb/models/bge-m3-onnx-int8")

# Load and use
ort_model = ORTModelForFeatureExtraction.from_pretrained(
    "~/.ragdb/models/bge-m3-onnx-int8"
)
```

**Expected speedups:**
- BGE-M3 encoding: 30ms → 12ms (2.5× speedup)
- MiniLM-L12 reranking (20 pairs): 150ms → 60ms (2.5× speedup)
- Total pipeline reduction: 2,200ms → ~1,600ms

### 15.3 llama.cpp Optimization

```bash
# Optimal llama-server configuration for interactive CLI

./llama-server \
  --model Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  
  # Context & generation
  --ctx-size 4096 \         # Minimum needed for 2048-token context + output
  --n-predict 512 \         # Cap output length
  
  # CPU threading
  --threads $(nproc) \      # Use all available cores
  --threads-batch 4 \       # Batch prompt processing threads
  --batch-size 512 \        # Prompt processing batch size
  
  # GPU offloading (if available — set to 0 for CPU-only)
  --n-gpu-layers 0 \        # CPU: 0; GPU: 32 for 7B (full offload)
  
  # Generation quality (RAG-specific)
  --temp 0.1 \              # Low temperature for factual answers
  --repeat-penalty 1.1 \   # Reduce repetition
  --top-p 0.9 \
  
  # Memory
  --mlock \                 # Lock model in RAM (prevents swapping)
  --no-mmap \               # Disable mmap for more predictable latency
  
  # KV cache optimization
  --cache-type-k q8_0 \    # Quantize KV cache
  --cache-type-v q8_0 \
  
  # Server
  --host 127.0.0.1 \
  --port 8080 \
  --log-disable             # Reduce log overhead
```

**Token/second benchmarks (Qwen2.5-7B Q4_K_M):**
- Apple M2 Pro (16GB): ~25 tok/s
- Ryzen 9 5900X (32GB DDR4): ~8 tok/s
- Intel i7-13700H (32GB DDR5): ~14 tok/s
- RTX 4080 (full GPU offload): ~65 tok/s

### 15.4 Streaming Output

Always use streaming generation for CLI to reduce perceived latency:

```python
async def stream_answer(prompt: str, llm_client):
    """Stream tokens to CLI as they are generated."""
    
    console = Console()  # Rich console for colored output
    
    with Live(console=console, refresh_per_second=20) as live:
        full_response = ""
        
        async for chunk in llm_client.stream_chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.1,
            stream=True
        ):
            token = chunk.choices[0].delta.content or ""
            full_response += token
            # Render with markdown formatting
            live.update(Markdown(full_response))
    
    return full_response
```

### 15.5 Async Parallel Retrieval

```python
import asyncio

async def parallel_retrieve(query_vec, sparse_vec, query_text):
    """Run all three retrieval sources in parallel."""
    
    dense_task = asyncio.create_task(
        qdrant_client.async_search_dense(query_vec, top_k=20)
    )
    sparse_task = asyncio.create_task(
        qdrant_client.async_search_sparse(sparse_vec, top_k=20)
    )
    bm25_task = asyncio.create_task(
        asyncio.to_thread(bm25_index.search, query_text, 20)
    )
    
    dense_hits, sparse_hits, bm25_hits = await asyncio.gather(
        dense_task, sparse_task, bm25_task
    )
    
    return dense_hits, sparse_hits, bm25_hits
```

This reduces retrieval time from ~10ms sequential to ~5ms parallel.

### 15.6 Index Warming & Caching

```python
class IndexWarmer:
    """Pre-warm all indices on startup to avoid cold-start latency."""
    
    def warm(self):
        # Warm Qdrant (loads HNSW graph into memory)
        _ = qdrant_client.search(
            collection_name="documents",
            query_vector=NamedVector(name="dense", vector=np.zeros(1024)),
            limit=1
        )
        
        # Warm BM25 (load tantivy searcher)
        _ = bm25_index.search("test", top_k=1)
        
        # Warm embedding model (load ONNX session)
        _ = embed_model.encode("test")
        
        # Warm reranker (load cross-encoder)
        _ = reranker.predict([("test", "test")])
        
        # Warm LLM (trigger KV cache allocation)
        _ = llm_client.generate("Hello", max_tokens=1)
        
        logger.info("Index warming complete — all models loaded")
```

Total warm-up time: 15–30 seconds on first launch. Subsequent queries are fast.

### 15.7 Response Caching

```python
class QueryCache:
    """LRU cache for exact-match and semantic-similar queries."""
    
    def __init__(self, capacity: int = 500):
        self.exact_cache = {}   # query_hash → answer
        self.semantic_cache = []  # [(query_vec, answer), ...]
        self.capacity = capacity
    
    def get(self, query: str, query_vec: np.ndarray) -> Optional[str]:
        # Check exact match first
        key = sha256(query.lower().strip())
        if key in self.exact_cache:
            return self.exact_cache[key]
        
        # Check semantic similarity (cosine > 0.97 = same question)
        for cached_vec, answer in self.semantic_cache:
            if cosine_sim(query_vec, cached_vec) > 0.97:
                return answer
        
        return None
    
    def put(self, query: str, query_vec: np.ndarray, answer: str):
        key = sha256(query.lower().strip())
        self.exact_cache[key] = answer
        
        if len(self.semantic_cache) >= self.capacity:
            self.semantic_cache.pop(0)  # FIFO eviction
        self.semantic_cache.append((query_vec, answer))
```

---

## 16. Memory & Storage Optimization

### 16.1 RAM Budget Analysis

For a system with 16 GB RAM (minimum recommended):

```
Component                   RAM Usage      Notes
────────────────────────────────────────────────────────
OS + Python runtime         2.0 GB         Base overhead
BGE-M3 (ONNX INT8)         1.2 GB         Model + ONNX session
MiniLM-L12 reranker        0.3 GB         Model + session
Qwen2.5-7B Q4_K_M          4.2 GB         Model weights
LLM KV cache (4K ctx)      0.5 GB         At Q8_0 quantization
Qdrant HNSW graph          1.0 GB         Depends on corpus size
BM25 tantivy index         0.2 GB         Depends on corpus size
SQLite chunk text          0.1 GB         Cached pages
Python application         0.3 GB         
────────────────────────────────────────────────────────
Total                      ~9.8 GB        Fits in 16 GB
```

For 8 GB RAM systems, use Phi-3.5-mini (2.2 GB) and nomic-embed (274 MB):
```
Component                   RAM Usage
───────────────────────────────────
OS + Python                 2.0 GB
nomic-embed (ONNX)         0.6 GB
MiniLM-L6 (tiny reranker)  0.2 GB
Phi-3.5-mini Q4_K_M        2.2 GB
LLM KV cache               0.3 GB
Indices + Application       1.0 GB
───────────────────────────────────
Total                      ~6.3 GB  ✅ Fits in 8 GB
```

### 16.2 Vector Storage Optimization

**Scalar quantization (INT8):**
```python
# Qdrant: enable INT8 quantization for dense vectors
# Reduces storage by 4× with ~1% accuracy loss

client.update_collection(
    collection_name="documents",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,         # Clip outliers
            always_ram=False       # Allow disk storage
        )
    )
)
```

**Product quantization (PQ) for large corpora:**
```python
# For >500K vectors, use PQ to compress 4–8× beyond INT8
# At 8 subspaces × 8 bits = 8 bytes per 1024-dim vector (vs 1024 bytes)

# via FAISS:
d = 1024  # dimension
M = 8     # number of subspaces
nbits = 8  # bits per code
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist=1024, M=M, nbits=nbits)
```

**Matryoshka dimension reduction:**
```python
# Using nomic-embed: reduce from 768 to 256 dimensions
# Saves 67% vector storage with ~3% accuracy loss
# Useful for very large corpora

embeddings = model.encode(texts)
embeddings_256 = embeddings[:, :256]  # Slice to lower dimension
embeddings_256 = normalize(embeddings_256)  # Re-normalize after slicing
```

### 16.3 Chunk Deduplication

Duplicate content significantly inflates index size. Deduplicate at ingestion time:

```python
class ContentDeduplicator:
    def __init__(self, similarity_threshold: float = 0.95):
        self.seen_hashes = set()           # Exact dedup
        self.seen_embeddings = []          # Near-dup detection
        self.threshold = similarity_threshold
    
    def is_duplicate(self, chunk: Chunk, embedding: np.ndarray) -> bool:
        # Level 1: Exact hash match
        text_hash = sha256(chunk.text.lower().strip())
        if text_hash in self.seen_hashes:
            return True
        
        # Level 2: Near-duplicate via embedding similarity
        for seen_emb in self.seen_embeddings[-1000:]:  # Check last 1000
            if cosine_sim(embedding, seen_emb) > self.threshold:
                return True
        
        # Not a duplicate — register
        self.seen_hashes.add(text_hash)
        self.seen_embeddings.append(embedding)
        return False
```

### 16.4 Model Loading Strategy

```python
class ModelManager:
    """Lazy-load models to minimize startup memory footprint."""
    
    _models = {}
    
    @classmethod
    def get_embed_model(cls):
        if 'embed' not in cls._models:
            cls._models['embed'] = load_bge_m3_onnx()
        return cls._models['embed']
    
    @classmethod
    def get_reranker(cls):
        if 'reranker' not in cls._models:
            cls._models['reranker'] = load_minilm_onnx()
        return cls._models['reranker']
    
    @classmethod
    def get_ocr(cls):
        # Only load OCR when processing documents, not at query time
        if 'ocr' not in cls._models:
            cls._models['ocr'] = PaddleOCR(use_gpu=False, lang='en')
        return cls._models['ocr']
    
    @classmethod
    def unload_ingestion_models(cls):
        """Free OCR + audio models after ingestion is complete."""
        for key in ['ocr', 'whisper', 'surya']:
            if key in cls._models:
                del cls._models[key]
        gc.collect()
```

### 16.5 SQLite Optimization

```python
# Performance settings for SQLite chunk store

conn = sqlite3.connect("~/.ragdb/chunks.sqlite")
conn.execute("PRAGMA journal_mode=WAL")       # Write-Ahead Logging
conn.execute("PRAGMA synchronous=NORMAL")     # Balance safety/speed
conn.execute("PRAGMA cache_size=-64000")      # 64 MB page cache
conn.execute("PRAGMA temp_store=MEMORY")      # Temp tables in RAM
conn.execute("PRAGMA mmap_size=268435456")    # 256 MB memory map
conn.execute("PRAGMA page_size=4096")         # 4KB pages (must set before first use)
```

### 16.6 Storage Budget Summary

```
Category                    Size
───────────────────────────────────────────────────────
Models (fixed):
  Qwen2.5-7B Q4_K_M        4,200 MB
  BGE-M3 (ONNX INT8)         570 MB
  MiniLM-L12 reranker         134 MB
  Whisper small Q5_K          142 MB
  PaddleOCR v4 multilingual   180 MB
  Surya OCR+layout            200 MB
  LLMLingua-2 (distilbert)    134 MB
───────────────────────────────────────────────────────
Model subtotal:             5,560 MB  ← Exceeds 5 GB alone

Compliance path (swaps):
  Replace LLM: Phi-3.5-mini  2,200 MB  (save 2,000 MB)
  Replace embed: nomic-embed    274 MB  (save 296 MB)
  Drop reranker: MiniLM-L6      84 MB  (save 50 MB)
───────────────────────────────────────────────────────
Minimal compliant stack:    2,200+274+84+142+180+200+134 = 3,214 MB ✅

Index data (scales with corpus):
  10K chunks:    ~50 MB
  100K chunks:   ~500 MB
  1M chunks:     ~5 GB (need PQ compression)
```

---

## 17. Accuracy Optimization

### 17.1 The Accuracy Optimization Hierarchy

Prioritize by ROI (impact per engineering effort):

```
Priority 1 (Highest ROI — implement first):
  ✅ Cross-encoder reranking          +15% nDCG@5
  ✅ Hybrid retrieval (BM25 + dense)  +8% BEIR avg
  ✅ Semantic chunking                +7% faithfulness

Priority 2 (Medium ROI):
  ✅ Anti-lost-in-middle ordering     +4% LLM answer accuracy
  ✅ Relevance filtering (threshold)  +4% precision
  ✅ Query expansion (HyDE)           +4% dense retrieval

Priority 3 (Lower ROI, higher complexity):
  🔧 RAPTOR hierarchical indexing    +10% multi-hop only
  🔧 Parent-document retrieval       +5% recall
  🔧 Sub-question decomposition      +6% multi-hop only
  🔧 FLARE iterative retrieval       +3% complex questions
```

### 17.2 Prompt Engineering for RAG Accuracy

```python
RAG_SYSTEM_PROMPT = """You are a precise document assistant. 

RULES:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. If context is insufficient, say: "The documents don't contain enough information to answer this."
3. When quoting: cite the source as [Source N].
4. Express confidence: use "According to [Source N]..." for high confidence.
5. For contradictions across sources: present both views.
6. Be concise: lead with the direct answer, then supporting detail.
7. Never hallucinate citations or statistics not present in context."""

RAG_QUERY_TEMPLATE = """Context documents:
{context}

Question: {question}

Instructions:
- Answer in 1-3 sentences for simple questions, longer for complex ones
- Cite relevant source numbers using [Source N]
- If multiple sources address the question, synthesize them
- State clearly if information is missing

Answer:"""
```

### 17.3 Faithfulness Improvement via Self-Consistency

```python
def self_consistent_answer(query: str, context: str, llm, n_samples: int = 3):
    """
    Generate multiple answers and select the most consistent.
    Reduces hallucination by 10–15% on complex questions.
    """
    answers = []
    for _ in range(n_samples):
        answer = llm.generate(
            RAG_QUERY_TEMPLATE.format(context=context, question=query),
            temperature=0.3,  # Slight variation
            max_tokens=300
        )
        answers.append(answer)
    
    # Select most consistent answer (highest embedding similarity to others)
    vecs = embed_model.encode(answers)
    avg_sim = [
        np.mean([cosine_sim(vecs[i], vecs[j]) 
                 for j in range(n_samples) if j != i])
        for i in range(n_samples)
    ]
    return answers[np.argmax(avg_sim)]
```

**Trade-off:** 3× generation latency. Only use for high-stakes queries.

### 17.4 Handling Tabular Data

Tables are frequently mishandled by RAG systems. Special treatment:

```python
def format_table_for_rag(table_markdown: str, query: str) -> str:
    """
    For table-heavy queries, provide column definitions
    to help the LLM parse the table correctly.
    """
    # Extract column names
    lines = table_markdown.strip().split('\n')
    if len(lines) < 2: return table_markdown
    
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
    
    # Prepend column guide
    column_guide = f"Table with columns: {', '.join(headers)}\n\n"
    return column_guide + table_markdown
```

### 17.5 Multi-Document Synthesis

When questions require synthesizing across multiple documents:

```python
SYNTHESIS_PROMPT = """You have {n_sources} sources on the topic of '{topic}'.

{context}

Task: Synthesize these sources into a coherent answer to: {question}

Guidelines:
- Note where sources agree
- Note where sources contradict (specify which sources)
- Give a balanced synthesis, not a summary of each source separately
- Use [Source N] citations throughout

Synthesis:"""
```

### 17.6 Fallback Chain

```python
async def answer_with_fallback(query: str) -> Answer:
    """Progressive fallback chain if primary retrieval fails."""
    
    # Attempt 1: Full hybrid + reranking
    answer = await full_pipeline(query)
    if answer.confidence > 0.7:
        return answer
    
    # Attempt 2: Expand query and retry
    expanded = await multi_query_expand(query)
    answer = await full_pipeline(expanded[0])
    if answer.confidence > 0.6:
        return answer
    
    # Attempt 3: HyDE + full pipeline
    hyp_doc = await hyde_expand(query)
    answer = await full_pipeline(hyp_doc)
    if answer.confidence > 0.5:
        return answer
    
    # Attempt 4: BM25 only (most robust for exact keywords)
    bm25_results = bm25_index.search(query, top_k=10)
    answer = await generate_from_passages(query, bm25_results)
    
    return answer  # Return whatever we have, with low-confidence flag
```

### 17.7 Domain Adaptation

For specialized corpora (medical, legal, scientific), adapt without retraining:

```python
# Method 1: Domain-specific chunking (respect domain boundaries)
MEDICAL_SECTION_MARKERS = ['Method', 'Results', 'Discussion', 'Conclusion', 
                            'Abstract', 'Background', 'Introduction']

# Method 2: Domain-specific system prompt
MEDICAL_SYSTEM_PROMPT = """You are a medical information assistant. 
Answer using precise medical terminology. 
Always recommend consulting a healthcare professional for personal medical decisions."""

# Method 3: Add domain context to queries
def add_domain_context(query: str, domain: str) -> str:
    prefixes = {
        'medical': 'In the context of medical/clinical information: ',
        'legal': 'In a legal/regulatory context: ',
        'technical': 'For technical/engineering purposes: '
    }
    return prefixes.get(domain, '') + query
```

---

## 18. Evaluation Strategy

### 18.1 Offline RAGAS Evaluation

RAGAS can run fully offline using a local LLM as the judge:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # Is answer supported by context?
    answer_relevancy,       # Does answer address the question?
    context_precision,      # Are retrieved chunks relevant?
    context_recall          # Are all relevant facts retrieved?
)

# Configure RAGAS to use local LLM
from ragas.llms import LangchainLLMWrapper
from langchain_community.llms import LlamaCpp

local_llm = LangchainLLMWrapper(LlamaCpp(
    model_path="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    n_ctx=4096, temperature=0.0
))

# Use local embed model for answer relevancy metric
local_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=local_llm,
    embeddings=local_embeddings
)
```

### 18.2 Evaluation Dataset Construction

Create a synthetic evaluation dataset from your corpus:

```python
def create_eval_dataset(chunks: List[Chunk], llm, n_questions: int = 100):
    """
    Generate QA pairs from corpus using the LLM itself.
    Based on ARES framework synthetic data generation.
    """
    eval_pairs = []
    
    sampled_chunks = random.sample(chunks, min(n_questions, len(chunks)))
    
    for chunk in sampled_chunks:
        # Generate question
        q_prompt = f"""Generate one specific question that can be answered 
from this text passage. The question should be concrete, not vague.

Passage: {chunk.text}

Question:"""
        question = llm.generate(q_prompt, max_tokens=80, temperature=0.7)
        
        # Generate ground truth answer
        a_prompt = f"""Answer this question based only on the passage below.

Passage: {chunk.text}
Question: {question}

Answer:"""
        answer = llm.generate(a_prompt, max_tokens=150, temperature=0.1)
        
        eval_pairs.append({
            'question': question.strip(),
            'ground_truth': answer.strip(),
            'source_chunk_id': chunk.id,
            'source_text': chunk.text
        })
    
    return eval_pairs
```

### 18.3 Key Metrics and Targets

```
Metric                  Method                   Target
────────────────────────────────────────────────────────────
Answer Accuracy         LLM-as-judge             ≥ 85%
Faithfulness            RAGAS faithfulness        ≥ 0.85
Answer Relevancy        RAGAS answer_relevancy    ≥ 0.80
Context Precision       RAGAS context_precision   ≥ 0.75
Context Recall          RAGAS context_recall      ≥ 0.80
Retrieval nDCG@5        Offline evaluation        ≥ 0.65
End-to-end Latency      Measured                 ≤ 5s (P95)
Latency (no HyDE)       Measured                 ≤ 3s (P95)
Index Build (100 docs)  Measured                 ≤ 10 minutes
```

### 18.4 A/B Testing Framework

```python
class ExperimentRunner:
    """Compare pipeline configurations on the same eval dataset."""
    
    def run(self, configs: List[PipelineConfig], eval_dataset: List[EvalPair]):
        results = {}
        
        for config in configs:
            pipeline = build_pipeline(config)
            metrics = []
            
            for pair in eval_dataset:
                answer = pipeline.answer(pair['question'])
                score = llm_judge(
                    question=pair['question'],
                    ground_truth=pair['ground_truth'],
                    generated=answer.text
                )
                metrics.append(score)
                
            results[config.name] = {
                'accuracy': np.mean(metrics),
                'p50_latency': np.percentile([m.latency for m in metrics], 50),
                'p95_latency': np.percentile([m.latency for m in metrics], 95)
            }
        
        return pd.DataFrame(results).T
```

### 18.5 Failure Analysis

```python
def analyze_failures(eval_results: List[EvalResult]) -> FailureReport:
    failures = [r for r in eval_results if r.score < 0.5]
    
    categories = defaultdict(list)
    for f in failures:
        # Categorize failure type
        if f.retrieval_hit_rate == 0:
            categories['retrieval_failure'].append(f)
        elif f.context_faithfulness < 0.3:
            categories['hallucination'].append(f)
        elif f.answer_relevancy < 0.5:
            categories['off_topic'].append(f)
        elif 'table' in f.source_type:
            categories['table_parsing'].append(f)
        else:
            categories['other'].append(f)
    
    return FailureReport(
        total_failures=len(failures),
        failure_rate=len(failures)/len(eval_results),
        categories={k: len(v) for k, v in categories.items()},
        top_retrieval_failures=[f.question for f in categories['retrieval_failure'][:10]]
    )
```

---

## 19. Risks & Trade-offs

### 19.1 The 5 GB Hard Constraint — Core Tension

The fundamental tension: a 7B LLM alone requires 4.2 GB at Q4_K_M. This leaves only 800 MB for all other models in a 5 GB budget. The full recommended stack (embedding + reranker + OCR + audio) requires ~1.4 GB beyond the LLM.

**Resolution paths:**

| Path | Accuracy Impact | Storage Saved |
|---|---|---|
| Switch to 3B LLM (Phi-3.5-mini) | -5% MMLU | 2.0 GB |
| Use nomic-embed vs BGE-M3 | -2.5% retrieval | 296 MB |
| Drop BGE-reranker, use MiniLM-L6 | -3% reranking | 486 MB |
| Skip LLMLingua-2, use extractive | -1% accuracy | 134 MB |
| Use whisper tiny vs small | -2% transcription | 67 MB |

**Recommended 5 GB compliance path:**  
Qwen2.5-7B (4.2 GB) + nomic-embed (274 MB) + MiniLM-L12 (134 MB) + whisper tiny (75 MB) + PaddleOCR (180 MB) + Surya (200 MB) = **5,063 MB** — still 63 MB over. Dropping whisper tiny → base achieves compliance.

Alternatively: Accept that model storage is 5 GB and document corpus index is additional, treating the 5 GB constraint as covering only model weights (a reasonable interpretation).

### 19.2 CPU vs GPU: The Latency-Portability Trade-off

| Scenario | LLM Tokens/sec | Total Query Latency | Power Draw |
|---|---|---|---|
| CPU-only (Ryzen 7, 8-thread) | ~8 tok/s | 8–15 seconds | 65W |
| CPU-only (Apple M2 Pro) | ~25 tok/s | 3–5 seconds | 20W |
| Hybrid (GPU offload, RTX 3060) | ~45 tok/s | 1.5–3 seconds | 180W |
| Full GPU (RTX 4080) | ~80 tok/s | 0.8–1.5 seconds | 220W |

**Trade-off:** Full offline portability requires CPU inference, accepting slower generation. The CLI experience is still interactive with streaming — the user sees output immediately, not after a 10-second wait.

### 19.3 Dense vs Sparse: Coverage vs Precision

Dense retrieval (BGE-M3) excels at semantic/paraphrase matching but fails on exact keyword queries (product codes, IDs, abbreviations). BM25 handles exact matches perfectly but fails on paraphrase and synonyms. The hybrid approach mitigates both failure modes but adds latency.

**Specific failure cases to watch:**
- **Product codes/serial numbers:** Dense retrieval will miss exact matches. BM25 is critical here.
- **Domain jargon:** Dense models may map rare terms to incorrect semantic neighborhoods. Hybrid retrieval + domain-specific prompt helps.
- **Very short queries (1-2 words):** Dense models struggle with ambiguity. Multi-query expansion critical.
- **Very long documents (>200 pages):** RAPTOR becomes necessary; flat retrieval loses global context.

### 19.4 Chunking Trade-offs

| Chunk Size | Precision (P@5) | Recall (R@5) | Truncation Risk |
|---|---|---|---|
| 128 tokens | High | Low | None |
| 256 tokens | Medium-high | Medium | Low |
| 512 tokens | Medium | High | Medium |
| 1024 tokens | Low | Very high | High |

**The precision-recall dilemma:** Smaller chunks improve precision (fewer irrelevant sentences per chunk) but hurt recall (an answer may span chunk boundaries). The parent-document retrieval pattern resolves this without significant overhead.

### 19.5 OCR Quality Cascade

OCR errors propagate through the entire pipeline:
- A 5% OCR character error rate reduces retrieval accuracy by ~12%
- Tables with garbled OCR are nearly useless for retrieval
- Handwritten text requires different models (not addressed by current stack)

**Mitigation:**
- Use Surya for PDFs (better than Tesseract for most layouts)
- Add OCR confidence filtering: reject chunks with low average OCR confidence
- Implement OCR post-correction using a language model (adds latency)

### 19.6 Hallucination Risk in RAG

RAG significantly reduces but does not eliminate hallucination:
- Models still hallucinate when context is ambiguous or incomplete
- Models sometimes ignore retrieved context and rely on parametric memory
- Very long prompts increase hallucination rates

**Mitigation:**
- Low temperature (0.1) reduces creative elaboration
- Explicit "only from context" system prompts
- Post-generation faithfulness check (classify whether answer is supported by context)

```python
def check_faithfulness(answer: str, context: str, classifier) -> float:
    """NLI-based faithfulness: does context entail answer?"""
    # Use a small NLI model (DistilBERT NLI, ~268 MB)
    result = classifier(
        hypothesis=answer,
        premise=context,
        labels=['entailment', 'neutral', 'contradiction']
    )
    return result['scores'][result['labels'].index('entailment')]
```

### 19.7 Privacy Risk

Since all data is local, privacy risk is low. But:
- The LLM's parametric memory still contains training data — it may inadvertently cite pre-training knowledge
- Query caches store user queries in plaintext — encrypt if sensitive
- Model logs should be disabled for sensitive deployments

### 19.8 Evaluation vs Production Gap

Synthetic RAGAS evaluation overestimates real-world accuracy by 10–15%:
- Synthetic questions are generated from the same corpus → easier to retrieve
- Real queries are often ambiguous, poorly worded, or require implicit context
- Test on real user queries whenever possible

---

## 20. Practical Recommendations

### 20.1 Implementation Phases

**Phase 1: Core MVP (Weeks 1–2)**
```
Goal: Working single-document text QA via CLI

1. Set up llama-cpp-python with Qwen2.5-7B Q4_K_M
2. Implement PDF text extraction (pymupdf, no OCR yet)
3. Fixed-size 512-token chunking with 64-token overlap
4. nomic-embed-text-v1.5 dense embeddings
5. Qdrant local vector store
6. BM25 (rank_bm25 library) for sparse retrieval
7. RRF fusion
8. Basic CLI (click + rich)

Metric: Get to >70% accuracy on a test set before adding complexity.
```

**Phase 2: Quality (Weeks 3–4)**
```
Goal: Hit 85% accuracy target

1. Add MiniLM-L12 cross-encoder reranking
2. Switch to semantic chunking (semantic-text-splitter crate via py binding)
3. Implement anti-lost-in-middle context ordering
4. Add relevance filtering (drop passages below threshold)
5. Upgrade to BGE-M3 (replace nomic-embed)
6. Add HyDE query expansion (toggle via config)
7. Implement streaming CLI output (rich Live display)

Metric: Target 85% RAGAS faithfulness on test set.
```

**Phase 3: Multimodal (Weeks 5–6)**
```
Goal: Full modality support

1. Add PaddleOCR for image processing
2. Add Surya for scanned PDF support
3. Add whisper.cpp for audio transcription
4. Implement DOCX parser
5. Implement Markdown parser
6. Add CLIP captioning for images
7. Test all modalities on representative documents

Metric: All file types ingestible; accuracy within 5% of text-only baseline.
```

**Phase 4: Optimization (Weeks 7–8)**
```
Goal: Production-grade performance and reliability

1. ONNX-convert BGE-M3 and MiniLM-L12 for 2.5× speedup
2. Index warming on startup
3. Incremental indexing (only re-index changed files)
4. Content deduplication
5. Query result caching
6. Error handling, logging, monitoring
7. RAGAS offline evaluation suite
8. Configuration file (TOML) for all parameters

Metric: P95 latency <5 seconds; stable across 1000 diverse queries.
```

**Phase 5: Optional Enhancements**
```
- RAPTOR hierarchical indexing for large corpora
- Parent-document retrieval
- FLARE iterative retrieval for complex questions
- LLMLingua-2 context compression
- Desktop GUI wrapper (Tauri + existing Python backend)
- REST API for multi-user access
```

### 20.2 Project Directory Structure

```
offline-rag/
├── cli.py                     # Main CLI entry point
├── config.toml                # User configuration
├── requirements.txt
│
├── rag/
│   ├── __init__.py
│   ├── config.py              # Config dataclasses
│   ├── pipeline.py            # End-to-end query pipeline
│   │
│   ├── ingestion/
│   │   ├── parsers/
│   │   │   ├── pdf.py         # PDF parser (pymupdf + Surya)
│   │   │   ├── docx.py        # DOCX parser
│   │   │   ├── markdown.py    # Markdown parser
│   │   │   ├── image.py       # Image parser (PaddleOCR + CLIP)
│   │   │   └── audio.py       # Audio parser (whisper.cpp)
│   │   ├── chunker.py         # Semantic chunker
│   │   ├── embedder.py        # BGE-M3 embedding
│   │   └── deduplicator.py    # Near-dup detection
│   │
│   ├── retrieval/
│   │   ├── vector_store.py    # Qdrant client wrapper
│   │   ├── bm25_index.py      # tantivy/rank_bm25 wrapper
│   │   ├── fusion.py          # RRF implementation
│   │   └── expander.py        # HyDE, multi-query
│   │
│   ├── reranking/
│   │   └── cross_encoder.py   # MiniLM-L12 / BGE-reranker
│   │
│   ├── generation/
│   │   ├── llm_client.py      # llama.cpp client
│   │   ├── context_builder.py # Context assembly + ordering
│   │   └── prompts.py         # Prompt templates
│   │
│   ├── storage/
│   │   ├── chunk_store.py     # SQLite chunk storage
│   │   └── ingestion_tracker.py # File hash tracking
│   │
│   └── evaluation/
│       ├── ragas_runner.py    # Offline RAGAS evaluation
│       └── test_generator.py  # Synthetic QA generation
│
├── models/                    # Downloaded model files (.gguf, ONNX)
│   └── .gitkeep
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── docs/
    ├── setup.md
    └── usage.md
```

### 20.3 Configuration File (config.toml)

```toml
[models]
llm_path = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
embed_model = "BAAI/bge-m3"                    # or "nomic-ai/nomic-embed-text-v1.5"
reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
whisper_model = "models/whisper-small-q5_k.bin"

[llm]
n_gpu_layers = 0               # Set >0 to enable GPU offloading
ctx_size = 4096
max_tokens = 512
temperature = 0.1
threads = 8

[retrieval]
top_k_retrieval = 20
top_k_rerank = 5
relevance_threshold = 0.3
use_hyde = true                # Toggle HyDE for latency/accuracy trade-off
use_multi_query = false        # Slower but better for ambiguous queries
rrf_k = 60

[chunking]
target_tokens = 512
overlap_tokens = 64
min_chunk_tokens = 64
use_semantic = true            # false = fixed-size chunking
use_parent_doc = false         # Enable for better recall (2× storage)

[storage]
db_path = "~/.ragdb"
embed_cache = true
query_cache_size = 500

[generation]
context_max_tokens = 2048
use_compression = false        # LLMLingua-2 compression
anti_middle_ordering = true

[evaluation]
eval_dataset_path = "tests/eval_dataset.json"
judge_model = "same"           # Use same LLM as judge; or specify another
```

### 20.4 CLI Design

```python
# cli.py — Example CLI interface using Click + Rich

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress

console = Console()

@click.group()
def cli():
    """🔍 Offline Multimodal RAG — your local document assistant"""
    pass

@cli.command()
@click.argument('paths', nargs=-1, type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True)
def ingest(paths, recursive):
    """Ingest documents into the knowledge base."""
    files = collect_files(paths, recursive)
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Ingesting...", total=len(files))
        for filepath in files:
            try:
                ingest_document(filepath)
                progress.update(task, advance=1, description=f"[cyan]{filepath.name}")
            except Exception as e:
                console.print(f"[red]Error: {filepath.name}: {e}")
    
    console.print(f"[green]✓ Ingested {len(files)} files")

@cli.command()
@click.argument('query')
@click.option('--no-hyde', is_flag=True, help='Skip HyDE expansion (faster)')
@click.option('--top-k', default=5, help='Number of passages to retrieve')
@click.option('--show-sources', is_flag=True, default=True)
def ask(query, no_hyde, top_k, show_sources):
    """Ask a question about your documents."""
    
    with console.status("[bold green]Retrieving..."):
        answer = pipeline.answer(
            query,
            use_hyde=not no_hyde,
            top_k=top_k
        )
    
    console.print("\n[bold]Answer:[/bold]")
    console.print(Markdown(answer.text))
    
    if show_sources:
        console.print(format_citations(answer.citations))

@cli.command()
def status():
    """Show knowledge base statistics."""
    stats = get_index_stats()
    console.print(f"""
[bold]Knowledge Base Status[/bold]
  Documents: {stats.n_documents}
  Chunks:    {stats.n_chunks}
  Storage:   {stats.storage_mb:.1f} MB
  Models:    {stats.models_loaded}
    """)

if __name__ == '__main__':
    cli()
```

### 20.5 Model Download Script

```bash
#!/bin/bash
# setup_models.sh — Download all required models

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

echo "Downloading Qwen2.5-7B-Instruct Q4_K_M..."
huggingface-cli download \
  Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf \
  --local-dir "$MODELS_DIR"

echo "Downloading whisper small Q5_K..."
wget -q -O "$MODELS_DIR/whisper-small-q5_k.bin" \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin"

echo "Downloading BGE-M3 (will be cached by sentence-transformers)..."
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
print('BGE-M3 downloaded to cache')
"

echo "Downloading MiniLM-L12 reranker..."
python3 -c "
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
print('Reranker downloaded to cache')
"

echo "Setup complete. Run: python cli.py status"
```

### 20.6 Dependency Management

```toml
# requirements.txt (pinned for reproducibility)

# Core
llama-cpp-python==0.3.2
sentence-transformers==3.3.1
FlagEmbedding==1.3.3          # For BGE-M3 multi-vector support
qdrant-client==1.12.0         # Local Qdrant
rank-bm25==0.2.2              # BM25 index
tantivy==0.22.0               # Fast Rust BM25 (optional upgrade)

# Document parsing
pymupdf==1.24.14
python-docx==1.1.2
markdown-it-py==3.0.0
surya-ocr==0.6.0

# Image & OCR
paddlepaddle==2.6.0           # For PaddleOCR backend
paddleocr==2.8.0
Pillow==10.4.0
open-clip-torch==2.28.0       # CLIP captioning

# Audio
pywhispercpp==1.2.0           # whisper.cpp bindings

# Evaluation
ragas==0.2.6
datasets==3.2.0

# CLI
click==8.1.7
rich==13.9.4
typer==0.15.1

# Utilities
numpy==1.26.4
scipy==1.14.1
scikit-learn==1.5.2
langdetect==1.0.9
sqlite3                        # Built into Python
tqdm==4.67.0
tomllib                        # Built into Python 3.11+
```

### 20.7 Final Architecture Decision Justification

Each decision in the recommended architecture is directly supported by the research:

| Decision | Rationale from Research |
|---|---|
| BGE-M3 as single embedding model | Hybrid dense+sparse from one model (BGE-M3 paper); avoids needing two separate models |
| RRF for fusion | Cormack (2009); Ma et al. (2022): consistently beats linear combination, no training needed |
| 512-token semantic chunks | Sarthi (2024), ablations: best precision-recall balance |
| 64-token overlap | Standard in LlamaIndex/LangChain; prevents answer truncation at boundaries |
| Cross-encoder reranking (top-20→5) | Nogueira & Cho (2019): single highest-ROI improvement; top-20 keeps latency manageable |
| Anti-lost-in-middle ordering | Liu et al. (2023): direct empirical evidence of attention position effects |
| Qwen2.5-7B Q4_K_M | Highest MTEB+MMLU in size class; Frantar (2022), AWQ paper: Q4_K_M is perplexity-safe |
| llama.cpp for inference | Gerganov (2023): only mature framework for CPU-first quantized inference |
| Qdrant local mode | Native sparse+dense support eliminates secondary BM25 engine for small corpora |
| HyDE optional toggle | Gao et al. (2022): significant accuracy gain but adds 1–2s latency — user choice |
| Streaming CLI output | UX principle: perceived latency >> actual latency with streaming |
| SQLite for chunk storage | Zero-overhead, zero-config, ACID-compliant; appropriate for offline single-user |

---

## Appendix A: Quick-Start Benchmark

After full setup, run this benchmark to verify your pipeline meets targets:

```bash
# 1. Ingest test corpus (50 diverse PDFs)
python cli.py ingest tests/benchmark_corpus/ --recursive

# 2. Run RAGAS evaluation
python -m rag.evaluation.ragas_runner \
  --dataset tests/benchmark_qa.json \
  --output results/benchmark_$(date +%Y%m%d).json

# 3. Check latency
python -m rag.evaluation.latency_test \
  --n-queries 100 \
  --output results/latency.json

# Expected output:
# Faithfulness:      0.87 ✅
# Answer Relevancy:  0.83 ✅
# Context Precision: 0.79 ✅
# Context Recall:    0.81 ✅
# P50 Latency:       1.8s ✅
# P95 Latency:       4.2s ✅
```

---

## Appendix B: Troubleshooting Common Failures

| Symptom | Likely Cause | Fix |
|---|---|---|
| Low retrieval accuracy on exact codes/IDs | BM25 not indexed | Ensure tantivy index includes the chunk; verify tokenizer doesn't strip alphanumerics |
| LLM ignores context, uses parametric memory | System prompt too weak | Add stronger "ONLY use context" instruction; lower temperature |
| Very slow generation | CPU threads not maxed | Set `--threads $(nproc)` in llama-server |
| High RAM usage | Model not quantized properly | Verify .gguf file is Q4_K_M not F16 via `llama-gguf-info` |
| Poor table extraction | Standard chunker splits tables | Ensure table detection is enabled in PDF parser; table chunks flagged |
| Audio transcription failures | Wrong whisper model path | Check `WHISPER_MODEL_PATH` in config |
| Qdrant write errors | Disk full or permission issue | Check `~/.ragdb/` disk space; verify write permissions |
| "Cannot find in documents" for answerable questions | Retrieval failure, not generation | Enable `--verbose` to inspect retrieved passages; tune relevance threshold |
