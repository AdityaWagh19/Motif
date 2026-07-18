---

## 5. Recommended System Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OFFLINE MULTIMODAL RAG SYSTEM                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    INGESTION PIPELINE                         │   │
│  │                                                              │   │
│  │  PDF ──► Surya Layout+OCR ──┐                               │   │
│  │  DOCX ─► python-docx ───────┤                               │   │
│  │  MD ───► direct parse ──────┼──► Text Normalizer            │   │
│  │  IMG ──► PaddleOCR+CLIP ────┤        │                      │   │
│  │  WAV ──► whisper.cpp ───────┘        ▼                      │   │
│  │                               Semantic Chunker               │   │
│  │                               (512 tok / 64 overlap)         │   │
│  │                                      │                       │   │
│  │                          ┌───────────┼───────────┐           │   │
│  │                          ▼           ▼           ▼           │   │
│  │                     BGE-M3       BGE-M3       tantivy       │   │
│  │                    Dense Vec   Sparse Vec     BM25 Index    │   │
│  │                          │           │           │           │   │
│  │                          └─────┬─────┘           │           │   │
│  │                                ▼                 │           │   │
│  │                         Qdrant (local)           │           │   │
│  │                      [HNSW + Sparse idx]         │           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      QUERY PIPELINE                           │   │
│  │                                                              │   │
│  │  CLI Input ──► Query Analyzer ──► Query Expander (HyDE)     │   │
│  │                                          │                   │   │
│  │                              ┌───────────┼───────────┐       │   │
│  │                              ▼           ▼           ▼       │   │
│  │                         Qdrant Dense  Qdrant Sparse  BM25   │   │
│  │                          (top-20)      (top-20)    (top-20) │   │
│  │                              │           │           │       │   │
│  │                              └─────┬─────┘───────────┘       │   │
│  │                                    ▼                         │   │
│  │                               RRF Fusion                     │   │
│  │                                (top-20)                      │   │
│  │                                    │                         │   │
│  │                                    ▼                         │   │
│  │                         MiniLM-L12 Reranker                 │   │
│  │                               (top-5)                        │   │
│  │                                    │                         │   │
│  │                                    ▼                         │   │
│  │                      Context Constructor                     │   │
│  │                   (relevance-ordered + compressed)           │   │
│  │                                    │                         │   │
│  │                                    ▼                         │   │
│  │                    Qwen2.5-7B-Instruct Q4_K_M               │   │
│  │                          (llama.cpp server)                  │   │
│  │                                    │                         │   │
│  │                                    ▼                         │   │
│  │                      Answer + Citations ──► CLI              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     PERSISTENCE LAYER                         │   │
│  │                                                              │   │
│  │  ~/.ragdb/                                                   │   │
│  │  ├── qdrant/         (vector + sparse index)                 │   │
│  │  ├── tantivy_bm25/   (BM25 inverted index)                   │   │
│  │  ├── chunks.sqlite   (chunk text + metadata)                 │   │
│  │  └── ingestion.sqlite (file hashes, ingestion status)        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Module Responsibilities

**Ingestion Pipeline (one-time cost):**
- Modality-specific parsers extract raw text and layout metadata
- Text Normalizer standardizes whitespace, removes artifacts, detects language
- Semantic Chunker produces overlapping chunks with metadata (source file, page, section)
- Embedding module (BGE-M3) runs dense + sparse encoding per chunk
- Qdrant stores dense + sparse vectors; tantivy stores BM25 index
- SQLite stores raw chunk text and metadata for retrieval at query time

**Query Pipeline (per-query):**
- CLI input → query analyzer (detect type: factual, comparative, multi-hop)
- Optional HyDE expansion via local LLM (on/off toggle for latency control)
- Parallel retrieval from all three indices
- RRF fusion → top-20 candidates
- MiniLM-L12 reranker → top-5 passages
- Context constructor assembles prompt (relevance order, not document order)
- llama.cpp server generates answer with streaming output to CLI
- Source citations extracted from metadata and displayed

**Persistence Layer:**
- All data stored locally in ~/.ragdb/ or user-configured path
- Zero network calls during normal operation
- Optional export of index to portable format

### 5.3 Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| CLI framework | Python + Rich | Rich provides streaming output, progress bars, syntax highlighting |
| Embedding inference | sentence-transformers + ONNX | CPU-optimized ONNX runtime, 2× faster than PyTorch |
| Vector store | Qdrant (local mode) | Native sparse + dense support, no server required in local mode |
| BM25 index | BM25Okapi (rank_bm25) or tantivy-py | tantivy is Rust-based, 10× faster for large corpora |
| Reranker inference | cross-encoder (sentence-transformers) | ONNX runtime option for speed |
| LLM inference | llama.cpp Python bindings (llama-cpp-python) | CPU+GPU hybrid offloading |
| PDF parsing | pymupdf (fitz) + Surya | pymupdf for text; Surya for complex layouts |
| DOCX parsing | python-docx | Standard, lightweight |
| Image OCR | PaddleOCR | Best open-source accuracy |
| Audio transcription | whisper.cpp via pywhispercpp | C++ backend, much faster than Python Whisper |
| Context compressor | LLMLingua-2 (distilbert backbone) | 4× compression with <5% accuracy loss |
| Evaluation | RAGAS (offline LLM judge mode) | Automated quality metrics |
| Configuration | TOML + Python dataclasses | Human-readable, no dependencies |

---

## 6. End-to-End Retrieval Pipeline

### 6.1 Ingestion Flow (Detailed)

```python
# Pseudo-code for ingestion pipeline

def ingest_document(filepath: Path, config: RAGConfig):
    
    # Step 1: Route to modality parser
    parser = get_parser(filepath.suffix)  # PDF, DOCX, MD, image, audio
    
    # Step 2: Extract raw content
    extraction = parser.extract(filepath)
    # Returns: List[Block] where Block = {text, type, page, bbox, metadata}
    
    # Step 3: Normalize text
    text_blocks = normalize(extraction.blocks)
    # - Remove headers/footers via heuristics
    # - Fix hyphenation
    # - Normalize whitespace
    # - Detect language (langdetect)
    # - Filter empty/boilerplate blocks
    
    # Step 4: Semantic chunking
    chunks = semantic_chunker.chunk(
        text_blocks,
        target_size=512,         # tokens
        overlap=64,              # tokens
        respect_sentences=True,  # never break mid-sentence
        metadata_inheritance=True # chunks inherit source metadata
    )
    
    # Step 5: Embed chunks (batched for efficiency)
    dense_vectors = embed_model.encode_batch(
        [c.text for c in chunks],
        batch_size=32,
        show_progress=True
    )
    sparse_vectors = embed_model.encode_sparse_batch(
        [c.text for c in chunks]
    )
    
    # Step 6: Persist
    chunk_ids = sqlite_store.insert_chunks(chunks)
    qdrant_client.upsert_vectors(chunk_ids, dense_vectors, sparse_vectors)
    bm25_index.add_documents(chunk_ids, [c.text for c in chunks])
    ingestion_tracker.mark_complete(filepath)
```

### 6.2 Query Flow (Detailed)

```python
def answer_query(query: str, config: RAGConfig) -> Answer:
    
    # Step 1: Query analysis
    query_type = classify_query(query)
    # Types: FACTUAL, COMPARISON, MULTI_HOP, GENERATIVE, UNKNOWN
    
    # Step 2: Query expansion (optional, config-controlled)
    expanded_queries = [query]
    if config.use_hyde and query_type in (FACTUAL, MULTI_HOP):
        hyp_doc = llm.generate(HYDE_PROMPT.format(query=query), max_tokens=200)
        expanded_queries.append(hyp_doc)
    
    # Step 3: Parallel retrieval
    all_results = []
    for q in expanded_queries:
        q_dense = embed_model.encode_query(q)
        q_sparse = embed_model.encode_sparse_query(q)
        
        dense_hits = qdrant.search_dense(q_dense, top_k=20)
        sparse_hits = qdrant.search_sparse(q_sparse, top_k=20)
        bm25_hits = bm25_index.search(q, top_k=20)
        all_results.extend([dense_hits, sparse_hits, bm25_hits])
    
    # Step 4: RRF fusion
    candidates = rrf_fuse(all_results, k=60)[:20]  # top-20 after fusion
    
    # Step 5: Fetch chunk text from SQLite
    passages = sqlite_store.fetch_chunks([c.id for c in candidates])
    
    # Step 6: Cross-encoder reranking
    scores = reranker.score_pairs(
        [(query, p.text) for p in passages]
    )
    top5 = sorted(zip(passages, scores), key=lambda x: -x[1])[:5]
    
    # Step 7: Filter irrelevant passages (threshold)
    top5 = [(p, s) for p, s in top5 if s > config.relevance_threshold]
    
    # Step 8: Context construction (Lost-in-the-Middle ordering)
    context = construct_context(top5, max_tokens=2048)
    # Place rank-1 first, rank-2 last, rank-3,4,5 in middle
    
    # Step 9: LLM generation with streaming
    prompt = RAG_PROMPT.format(context=context, query=query)
    answer = llm.generate_stream(prompt, max_tokens=500)
    
    # Step 10: Extract and format citations
    citations = extract_citations(top5)
    
    return Answer(text=answer, citations=citations, passages_used=top5)
```

### 6.3 Prompt Templates

**RAG System Prompt:**
```
You are a precise document assistant. Answer questions based ONLY on the 
provided context. If the answer is not in the context, say "I cannot find 
this in the provided documents."

Format your answer:
1. Direct answer in 1-3 sentences
2. Supporting details if necessary
3. Mark any uncertainty with "According to [source]..."

Do not add information from your training data.
```

**RAG Query Prompt:**
```
Context:
---
{context}
---

Question: {query}

Answer based strictly on the context above:
```

**HyDE Prompt:**
```
Write a 2-3 sentence passage that would directly answer the following question.
Write as if it came from a reference document.

Question: {query}

Passage:
```

---

## 7. Model Recommendations

### 7.1 LLM: Qwen2.5-7B-Instruct

**Primary recommendation:** `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4.2 GB)

Rationale:
- Highest MMLU score (74.2%) in the 7B class — indicates strong instruction following and knowledge integration critical for RAG
- Best HumanEval (79.4%) — structural reasoning needed for multi-step documents
- Excellent long-context performance (32K native context window, reduced to 4K for speed)
- 128K vocabulary supports multilingual documents without tokenization artifacts
- Strong performance on RAG-specific benchmarks (ExpertQA, FRAMES)

**Configuration for llama.cpp:**
```bash
./llama-server \
  --model Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --ctx-size 4096 \
  --n-gpu-layers 0 \    # CPU-only; set to 32+ for GPU offload
  --threads 8 \
  --batch-size 512 \
  --rope-freq-base 1000000 \  # Qwen2.5 RoPE base
  --repeat-penalty 1.1 \
  --temperature 0.1 \    # Low temp for factual RAG
  --port 8080
```

**Fallback:** `Phi-3.5-mini-instruct-Q4_K_M.gguf` (2.2 GB)
- Saves 2.0 GB at cost of ~5% MMLU drop
- Excellent reasoning-per-parameter ratio
- Best choice under extreme memory constraints (4 GB RAM total)

**Second fallback:** `Qwen2.5-3B-Instruct-Q4_K_M.gguf` (1.9 GB)

### 7.2 Embedding Model: BGE-M3

**Primary recommendation:** `BAAI/bge-m3` (INT8 quantized, ~570 MB)

Key capabilities:
- Dense retrieval (1024-dim vectors, HNSW-indexed)
- Sparse retrieval (30K-dim vocabulary weights, inverted index)
- ColBERT multi-vector (token-level, optional)
- 100+ languages
- 8192 token maximum sequence length (handles long document sections)

**Usage:**
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, use_int8=True)

# During indexing
outputs = model.encode(
    sentences,
    batch_size=32,
    max_length=512,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False  # Toggle on for max accuracy if budget allows
)

# During query
q_outputs = model.encode(
    [query],
    return_dense=True,
    return_sparse=True
)
```

**Budget fallback:** `nomic-ai/nomic-embed-text-v1.5` (274 MB)
- 8192 token context
- Strong MTEB performance (62.3 avg)
- Matryoshka embeddings (can use 256-dim instead of 768 to save storage)

### 7.3 Reranker: ms-marco-MiniLM-L-12

**Primary recommendation (budget):** `cross-encoder/ms-marco-MiniLM-L-12-v2` (134 MB)

**Full accuracy recommendation:** `BAAI/bge-reranker-v2-m3` (570 MB, multilingual)

**Usage:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-12-v2',
    max_length=512,
    device='cpu'
)

scores = reranker.predict(
    [(query, passage.text) for passage in candidates]
)
```

### 7.4 Audio Model: whisper.cpp

**Recommendation:** whisper small model, Q5_K quantized (142 MB)
- 5% WER on English, sufficient for RAG transcription
- whisper.cpp runs 3–5× faster than Python Whisper
- Supports automatic language detection

**Tiny model** (75 MB) if storage is critical — accepts slightly lower transcription quality

### 7.5 OCR Stack

**PDF layout:** Surya (Python, ~200 MB models)
- Handles multi-column, tables, headers/footers
- Returns structured JSON with bounding boxes

**Image OCR:** PaddleOCR v4 (~180 MB for multilingual model)
- Outperforms Tesseract on most benchmarks
- Handles rotated, low-contrast, and noisy text

**Academic PDFs:** NOUGAT (250 MB) as optional module
- Converts scientific PDFs to Markdown preserving equations and tables
- Toggle on for document collections with heavy scientific content

---

## 8. Embedding Strategy

### 8.1 Document Embedding

All document chunks are embedded into three representations:

**Dense vectors** (1024-dim float32 → INT8 quantized for storage):
```python
# 1024-dim, stored as int8 → 1024 bytes per chunk
dense_vector = bge_m3.encode_dense(chunk_text)
```

**Sparse vectors** (BGE-M3 learned sparse, stored as SPLADE-style dict):
```python
# Only non-zero weights stored: avg ~100–200 non-zero entries per chunk
sparse_vector = bge_m3.encode_sparse(chunk_text)
# Format: {token_id: weight, ...}
```

**BM25 terms** (classical term frequencies):
```python
# Preprocessed: lowercase, stopword removal, stemming (optional)
bm25_index.add(chunk_id, tokenize(chunk_text))
```

### 8.2 Query Embedding

Queries are encoded with a query-specific prefix for BGE-M3:
```python
query_text = f"Represent this sentence for searching relevant passages: {raw_query}"
```

This instruction-based encoding aligns the query distribution with the passage distribution more accurately, particularly for asymmetric retrieval tasks (short query vs. long passage).

### 8.3 Matryoshka Embeddings (Fallback Strategy)

If using nomic-embed-text-v1.5, Matryoshka representation learning allows dimension reduction:

```python
# Full 768-dim for max accuracy
# 512-dim: saves 33% storage, ~1% accuracy loss
# 256-dim: saves 67% storage, ~3% accuracy loss
# 128-dim: saves 83% storage, ~8% accuracy loss

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')
embeddings = model.encode(texts, normalize=True)
# Slice to desired dim: embeddings[:, :256]
```

For a 1M-chunk corpus, dropping from 768 to 256 dimensions saves 1.5 GB of vector storage.

### 8.4 Embedding Cache

Cache embeddings aggressively:
```python
# SQLite-based embedding cache
# Key: SHA256(text + model_name + dim)
# Value: embedding bytes

class EmbeddingCache:
    def get_or_compute(self, text: str) -> np.ndarray:
        key = sha256(text + MODEL_NAME)
        cached = self.db.get(key)
        if cached: return np.frombuffer(cached, dtype=np.float32)
        emb = model.encode(text)
        self.db.set(key, emb.tobytes())
        return emb
```

Typical cache hit rate after initial ingestion: >95% for repeated document sections, 0% for novel queries. Cache the document embeddings permanently; query embeddings are not worth caching.

### 8.5 Batch Encoding Strategy

```python
# For large corpora, batch encoding is critical
# BGE-M3 throughput: ~100 chunks/second on CPU, ~1000/second on GPU

OPTIMAL_BATCH_SIZE = 32  # for CPU
MAX_SEQUENCE_LENGTH = 512  # tokens, for indexing (not max 8192 — too slow)

# For queries: single encoding, no batching needed
```

---

## 9. Chunking Strategy

### 9.1 Recommended: Semantic Chunking with Overlap

The recommended chunking approach uses sentence boundary detection combined with embedding-based topic shift detection:

```python
class SemanticChunker:
    def __init__(self, target_tokens=512, overlap_tokens=64, 
                 threshold=0.3, min_chunk_tokens=64):
        self.splitter = SentenceSplitter(chunk_size=target_tokens, 
                                          chunk_overlap=overlap_tokens)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold  # cosine distance for topic shift
    
    def chunk(self, text_blocks: List[Block]) -> List[Chunk]:
        # Phase 1: Sentence-boundary-aware initial split
        sentences = self._extract_sentences(text_blocks)
        
        # Phase 2: Group sentences into semantic units
        groups = self._group_by_topic(sentences)
        
        # Phase 3: Split large groups, merge small ones
        chunks = self._balance_chunks(groups, self.target_tokens)
        
        # Phase 4: Add overlap (carry forward last N tokens of previous chunk)
        chunks = self._add_overlap(chunks, self.overlap_tokens)
        
        return chunks
    
    def _group_by_topic(self, sentences):
        if len(sentences) < 2: return [sentences]
        
        embeddings = self.embed_model.encode([s.text for s in sentences])
        groups, current = [], [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Cosine distance between consecutive sentence embeddings
            dist = 1 - cosine_similarity(embeddings[i-1:i], embeddings[i:i+1])[0][0]
            if dist > self.threshold:
                groups.append(current)
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        groups.append(current)
        return groups
```

### 9.2 Modality-Specific Chunking

**PDF chunks:**
- Preserve page numbers in metadata
- Do not split across page boundaries when near the target size
- Mark table content separately (table chunks have `type="table"`)
- Headers/footers are excluded via heuristics (short lines at consistent vertical positions)

**DOCX chunks:**
- Split at heading boundaries first (Heading 1/2/3 = natural chunk boundaries)
- Tables are extracted as single chunks regardless of token count
- Lists are kept intact (never split a bulleted list across chunks)

**Markdown chunks:**
- Split at `##` and `###` header boundaries
- Code blocks are treated as atomic units (never split)
- Tables preserved as atomic chunks

**Image chunks:**
- One chunk per image, containing OCR text + CLIP-generated caption
- Chunk metadata includes image path for source retrieval
- Format: `[Image: {filename}]\n[Caption: {clip_caption}]\n[Text: {ocr_text}]`

**Audio chunks:**
- Whisper produces timestamped segments (~30-second windows)
- Concatenate segments into ~512-token chunks preserving timestamps
- Chunk metadata includes start/end timestamps for source citation

### 9.3 Parent-Document Retrieval Pattern

Implement as a storage optimization that adds recall:

```python
class ParentDocumentRetriever:
    """
    Index small child chunks (128 tokens) for precision.
    At retrieval time, return their parent chunks (512 tokens) for context.
    """
    
    def index(self, document: Document):
        # Create parent chunks (512 tokens)
        parent_chunks = self.parent_chunker.chunk(document)
        parent_ids = self.store.save_parents(parent_chunks)
        
        # Create child chunks (128 tokens) from each parent
        for parent_id, parent in zip(parent_ids, parent_chunks):
            children = self.child_chunker.chunk(parent)
            for child in children:
                child.parent_id = parent_id
            self.embed_and_index(children)
    
    def retrieve(self, query: str, top_k: int = 5):
        # Search child embeddings for precision
        child_hits = self.vector_store.search(query, top_k=top_k*3)
        
        # Return unique parent chunks for context richness
        parent_ids = list(dict.fromkeys(h.parent_id for h in child_hits))
        return self.store.fetch_parents(parent_ids[:top_k])
```

### 9.4 RAPTOR Integration (Large Corpus Path)

For document collections exceeding 500 pages, add RAPTOR summarization during indexing:

```python
def raptor_index(chunks: List[Chunk], llm, embed_model):
    # Level 0: Raw semantic chunks
    level0 = chunks
    
    # Level 1: Cluster L0 by embedding similarity, summarize clusters
    clusters = gaussian_mixture_cluster(
        embeddings=embed_model.encode(chunks),
        n_components='auto'  # BIC-selected
    )
    level1 = []
    for cluster in clusters:
        summary = llm.summarize(cluster.chunks, max_tokens=256)
        level1.append(Chunk(text=summary, level=1, children=cluster.chunk_ids))
    
    # Level 2: Summarize level-1 summaries (global overview)
    if len(level1) > 10:
        global_summary = llm.summarize(level1, max_tokens=512)
        level2 = [Chunk(text=global_summary, level=2)]
    
    # Index all levels together
    index_all([level0, level1, level2])
```

**RAPTOR trade-off:** Indexing time increases 3–5× due to LLM summarization. Only worthwhile for collections where multi-hop retrieval is frequent (e.g., large textbooks, multi-chapter reports).

### 9.5 Chunk Metadata Schema

```python
@dataclass
class ChunkMetadata:
    chunk_id: str          # UUID
    source_path: str       # Original file path
    source_type: str       # "pdf", "docx", "md", "image", "audio"
    page_number: int       # For PDFs; -1 for non-paged
    section_title: str     # Nearest heading above chunk
    char_start: int        # Character offset in document
    char_end: int          # Character offset in document
    token_count: int       # Actual token count
    language: str          # ISO 639-1 language code
    parent_chunk_id: str   # For parent-doc retrieval; None if flat
    raptor_level: int      # 0=raw, 1=summary, 2=global; 0 for flat
    ingestion_time: float  # Unix timestamp
    file_hash: str         # SHA256 of source file for dedup
```

---

## 10. Indexing Strategy

### 10.1 Vector Index: Qdrant (Local Mode)

Qdrant operates in fully local mode (no server process needed):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    NamedVector, NamedSparseVector, PointStruct
)

client = QdrantClient(path="~/.ragdb/qdrant")  # Fully local, no network

client.create_collection(
    collection_name="documents",
    vectors_config={
        "dense": VectorParams(
            size=1024,         # BGE-M3 dense dim
            distance=Distance.COSINE,
            on_disk=True,      # Store vectors on disk, not RAM
            hnsw_config=HnswConfigDiff(
                m=16,          # Graph connectivity (16 is standard)
                ef_construct=100,  # Build quality vs speed
                full_scan_threshold=10_000  # Below this, use exact search
            )
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(
                on_disk=True,  # Inverted index on disk
                full_scan_threshold=5_000
            )
        )
    }
)
```

**Critical Qdrant settings for offline use:**
- `on_disk=True` for both dense and sparse: reduces RAM usage significantly
- `quantization_config=ScalarQuantization(type=ScalarType.INT8)`: halves vector storage
- `hnsw_config.ef=128` at query time for the recall/speed trade-off

### 10.2 BM25 Index: tantivy-py

```python
import tantivy

schema_builder = tantivy.SchemaBuilder()
schema_builder.add_text_field("chunk_id", stored=True)
schema_builder.add_text_field("body", stored=False, tokenizer_name="en_stem")
schema_builder.add_integer_field("ingestion_time", stored=True)
schema = schema_builder.build()

index = tantivy.Index(schema, path="~/.ragdb/tantivy_bm25")
writer = index.writer(heap_size=50_000_000)  # 50MB write buffer

# Indexing
for chunk in chunks:
    doc = tantivy.Document(
        chunk_id=[chunk.id],
        body=[chunk.text]
    )
    writer.add_document(doc)
writer.commit()
```

**Query time:**
```python
searcher = index.searcher()
query = index.parse_query(query_text, ["body"])
results = searcher.search(query, limit=20).hits
```

### 10.3 Chunk Storage: SQLite

```python
# SQLite for chunk text and metadata (fast key-value lookups)
conn = sqlite3.connect("~/.ragdb/chunks.sqlite")
conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        source_path TEXT,
        page_number INTEGER,
        section_title TEXT,
        token_count INTEGER,
        metadata JSON
    )
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source_path)")
```

### 10.4 Index Size Estimation

For a corpus of N documents averaging P pages each:

```
Chunks = N × P × 2  (approx 2 chunks per page at 512 tokens)
Dense vectors = Chunks × 1024 × 1 byte (INT8) = Chunks KB
Sparse entries = Chunks × 150 × 8 bytes (avg non-zeros × id+weight)
BM25 index = ~10% of raw text size

Example: 1000 PDF documents, avg 20 pages each
  Chunks = 1000 × 20 × 2 = 40,000 chunks
  Dense storage = 40,000 × 1024 bytes = 40 MB
  Sparse storage = 40,000 × 150 × 8 bytes = 48 MB
  SQLite chunk text = 40,000 × 2KB avg = 80 MB
  BM25 tantivy = ~8 MB
  Total index: ~176 MB  ← Very manageable
```

### 10.5 Incremental Indexing

```python
def ingest_incremental(filepath: Path):
    """Only index new or modified files."""
    file_hash = sha256_file(filepath)
    
    existing = ingestion_db.get(filepath)
    if existing and existing.hash == file_hash:
        logger.info(f"Skipping {filepath} — unchanged")
        return
    
    if existing:
        # Delete old chunks
        old_chunk_ids = chunk_db.get_by_source(str(filepath))
        qdrant.delete_points(old_chunk_ids)
        bm25_index.delete_documents(old_chunk_ids)
        chunk_db.delete_by_source(str(filepath))
    
    # Re-index
    ingest_document(filepath)
    ingestion_db.upsert(filepath, file_hash, datetime.now())
```

---

## 11. Multimodal Processing Pipeline

### 11.1 PDF Processing Pipeline

```
PDF File
  │
  ├─► pymupdf text extraction (fast path for text-layer PDFs)
  │     Returns: TextBlocks with page + position info
  │     Quality check: if text_density < threshold → OCR path
  │
  ├─► Surya layout analysis (for complex/scanned PDFs)
  │     Returns: LayoutResult with reading order, columns, table regions
  │     Feeds into: Surya OCR for scanned regions
  │
  ├─► Table detection & extraction
  │     Tool: pymupdf table extraction or camelot-py
  │     Output: Markdown table strings embedded in text chunks
  │     Metadata: table=True flag for filtering
  │
  ├─► Figure detection
  │     Tool: pymupdf image extraction + PaddleOCR
  │     Output: OCR text + CLIP caption
  │     Storage: image saved to ~/.ragdb/images/{hash}.png
  │
  └─► Merge & order: Combine all blocks in reading order
        → Feed to chunker
```

**Key heuristics for PDF text quality:**
```python
def needs_ocr(page) -> bool:
    text = page.get_text()
    # Signs of a scanned PDF:
    if len(text.strip()) < 50 and page.get_image_list():
        return True
    # Signs of garbled text:
    if count_unicode_errors(text) / len(text) > 0.05:
        return True
    return False
```

### 11.2 DOCX Processing Pipeline

```python
from docx import Document
from docx.shared import Inches

def parse_docx(filepath: Path) -> List[Block]:
    doc = Document(filepath)
    blocks = []
    
    for element in doc.element.body:
        if element.tag.endswith('p'):  # Paragraph
            para = parse_paragraph(element, doc)
            if para.style.startswith('Heading'):
                blocks.append(Block(text=para.text, type='heading', 
                                    level=int(para.style[-1])))
            else:
                blocks.append(Block(text=para.text, type='paragraph'))
        
        elif element.tag.endswith('tbl'):  # Table
            table_md = extract_table_as_markdown(element, doc)
            blocks.append(Block(text=table_md, type='table'))
    
    # Extract embedded images
    for rel in doc.part.rels.values():
        if 'image' in rel.reltype:
            img_bytes = rel.target_part.blob
            ocr_text = paddleocr.ocr(img_bytes)
            caption = clip_captioner.caption(img_bytes)
            blocks.append(Block(
                text=f"[Image]\n{caption}\n{ocr_text}",
                type='image'
            ))
    
    return blocks
```

### 11.3 Image Processing Pipeline

```python
def process_image(filepath: Path) -> List[Block]:
    img = PIL.Image.open(filepath)
    
    # Preprocessing
    img = auto_rotate(img)       # EXIF-based rotation
    img = enhance_contrast(img)  # Histogram equalization for OCR
    
    # OCR
    ocr_result = paddleocr.ocr(np.array(img), cls=True)
    ocr_text = "\n".join([line[1][0] for box in ocr_result for line in box])
    
    # Visual understanding (CLIP caption)
    caption = clip_model.generate_caption(img)
    
    # Combine
    combined = f"[Image: {filepath.name}]\n"
    if caption:
        combined += f"Description: {caption}\n"
    if ocr_text:
        combined += f"Text content:\n{ocr_text}"
    
    return [Block(
        text=combined,
        type='image',
        metadata={'filepath': str(filepath), 'has_text': bool(ocr_text)}
    )]
```

### 11.4 Audio Processing Pipeline

```python
def process_audio(filepath: Path) -> List[Block]:
    # whisper.cpp via Python bindings
    model = Whisper(model_path="~/.ragdb/models/whisper-small-q5_k.bin")
    
    result = model.transcribe(
        str(filepath),
        language=None,      # Auto-detect
        word_timestamps=True,
        initial_prompt="This is a recorded meeting/lecture transcription:"
    )
    
    blocks = []
    current_text = ""
    current_start = 0.0
    
    for segment in result.segments:
        current_text += segment.text + " "
        
        # Create chunk when we hit target token count or silence gap
        if count_tokens(current_text) >= 450 or segment.has_silence_after:
            blocks.append(Block(
                text=current_text.strip(),
                type='audio_transcript',
                metadata={
                    'start_time': current_start,
                    'end_time': segment.end,
                    'language': result.language
                }
            ))
            current_text = ""
            current_start = segment.end
    
    if current_text:
        blocks.append(Block(text=current_text.strip(), type='audio_transcript'))
    
    return blocks
```

### 11.5 Markdown Processing Pipeline

```python
def process_markdown(filepath: Path) -> List[Block]:
    content = filepath.read_text(encoding='utf-8')
    
    # Parse with mistune or markdown-it-py
    md_parser = MarkdownParser()
    ast = md_parser.parse(content)
    
    blocks = []
    for node in ast:
        if node.type == 'heading':
            blocks.append(Block(
                text=node.text,
                type=f'heading_{node.level}',
                metadata={'level': node.level}
            ))
        elif node.type == 'code_block':
            blocks.append(Block(
                text=f"```{node.lang}\n{node.code}\n```",
                type='code',
                metadata={'language': node.lang}
            ))
        elif node.type == 'table':
            blocks.append(Block(
                text=node_to_markdown(node),
                type='table'
            ))
        else:
            blocks.append(Block(text=node.text, type='paragraph'))
    
    return blocks
```

### 11.6 Extensibility: Adding New Modalities

The parser is designed for extensibility via a registry pattern:

```python
PARSER_REGISTRY = {
    '.pdf':  PDFParser,
    '.docx': DOCXParser,
    '.doc':  LegacyDOCParser,  # converts to docx first
    '.md':   MarkdownParser,
    '.txt':  PlainTextParser,
    '.png':  ImageParser,
    '.jpg':  ImageParser,
    '.jpeg': ImageParser,
    '.tiff': ImageParser,
    '.wav':  AudioParser,
    '.mp3':  AudioParser,
    '.m4a':  AudioParser,
    '.ogg':  AudioParser,
    # Future: .pptx, .xlsx, .epub, .html
}

def get_parser(suffix: str) -> BaseParser:
    suffix = suffix.lower()
    if suffix not in PARSER_REGISTRY:
        raise UnsupportedFormatError(f"No parser for {suffix}")
    return PARSER_REGISTRY[suffix]()
```

---

## 12. Query Processing Pipeline

### 12.1 Query Analysis

```python
class QueryAnalyzer:
    """Classify query type to route to optimal retrieval strategy."""
    
    QUERY_PATTERNS = {
        'FACTUAL': [
            r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b',
            r'\bwhere is\b', r'\bhow many\b', r'\bdefine\b'
        ],
        'COMPARISON': [
            r'\bcompare\b', r'\bdifference between\b', r'\bvs\b',
            r'\bbetter\b.*\bworse\b', r'\bpros and cons\b'
        ],
        'MULTI_HOP': [
            r'\bbecause of\b', r'\bwhy\b.*\bbecause\b',
            r'\brelationship between\b', r'\bhow does.*affect\b'
        ],
        'GENERATIVE': [
            r'\bsummarize\b', r'\bexplain\b', r'\bdescribe\b',
            r'\bwrite\b', r'\blist\b', r'\boutline\b'
        ]
    }
    
    def analyze(self, query: str) -> QuerySpec:
        query_lower = query.lower()
        
        for qtype, patterns in self.QUERY_PATTERNS.items():
            if any(re.search(p, query_lower) for p in patterns):
                return QuerySpec(type=qtype, use_hyde=(qtype != 'FACTUAL'))
        
        return QuerySpec(type='UNKNOWN', use_hyde=False)
```

### 12.2 Query Expansion Strategies

**Strategy A: HyDE (Hypothetical Document Embedding)**
Best for: dense retrieval, topic-rich queries, multi-hop questions
```python
def hyde_expand(query: str, llm) -> str:
    prompt = f"""Write a 2-3 sentence passage that would directly answer:
"{query}"
Write as if from a reference document, not as an AI.
Passage:"""
    return llm.generate(prompt, max_tokens=150, temperature=0.3)
```

**Strategy B: Multi-Query Expansion**
Best for: comparative questions, broad topic retrieval
```python
def multi_query_expand(query: str, llm) -> List[str]:
    prompt = f"""Generate 3 different search queries for: "{query}"
Output exactly 3 queries, one per line, no numbering:"""
    response = llm.generate(prompt, max_tokens=100, temperature=0.5)
    return [q.strip() for q in response.split('\n') if q.strip()][:3]
```

**Strategy C: Sub-Question Decomposition**
Best for: multi-hop questions requiring multiple document lookups
```python
def decompose_query(query: str, llm) -> List[str]:
    prompt = f"""Break this question into 2-3 simpler sub-questions:
"{query}"
Sub-questions (one per line):"""
    response = llm.generate(prompt, max_tokens=150)
    sub_questions = [q.strip() for q in response.split('\n') if q.strip()]
    return sub_questions + [query]  # Include original
```

### 12.3 Retrieval Execution

```python
async def execute_retrieval(
    queries: List[str],
    embed_model,
    qdrant_client,
    bm25_index,
    top_k: int = 20
) -> List[ScoredPassage]:
    
    all_hits = defaultdict(list)
    
    for q in queries:
        # Dense query vector
        dense_vec = embed_model.encode_dense(q, query_instruction=True)
        
        # Sparse query vector  
        sparse_vec = embed_model.encode_sparse(q)
        
        # Run all three retrievals
        dense_results = qdrant_client.search(
            collection_name="documents",
            query_vector=NamedVector(name="dense", vector=dense_vec),
            limit=top_k
        )
        
        sparse_results = qdrant_client.search(
            collection_name="documents",
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(indices=sparse_vec.indices, 
                                    values=sparse_vec.values)
            ),
            limit=top_k
        )
        
        bm25_results = bm25_index.search(q, top_k=top_k)
        
        # Collect with rank information for RRF
        for rank, hit in enumerate(dense_results):
            all_hits[hit.id].append(('dense', rank + 1))
        for rank, hit in enumerate(sparse_results):
            all_hits[hit.id].append(('sparse', rank + 1))
        for rank, hit in enumerate(bm25_results):
            all_hits[hit.id].append(('bm25', rank + 1))
    
    # RRF fusion
    rrf_scores = {}
    for doc_id, rank_list in all_hits.items():
        rrf_scores[doc_id] = sum(1 / (60 + r) for _, r in rank_list)
    
    # Sort by RRF score, return top-k
    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
    return [ScoredPassage(id=doc_id, rrf_score=score) for doc_id, score in ranked]
```

