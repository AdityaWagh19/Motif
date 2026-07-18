# Pipeline Data Flows — Motif Offline Multimodal RAG

> **Depends on:** `architecture.md`  
> **Purpose:** Defines exactly how data moves through the system at each stage — the interface contracts that every module must satisfy.

---

## 0. Session Startup Flow

```
motif                          ← user runs the command
  │
  ├─└► load config.toml
  │     detect_hardware_tier() → "T1" | "T2" | "T3"
  │
  ├─└► Session.load()
  │     check ~/.ragdb/history.json
  │     if exists: load conversation_history list
  │     if not:    start with empty history []
  │
  ├─└► Print welcome screen (Rich panel)
  │     ┌──────────────────────────────────────────────┐
  │     │  Motif v0.1.0                                     │
  │     │  Tier: T2  |  Qwen2.5-7B Q4_K_M  |  GTX 1650    │
  │     │  Index: 3,241 chunks  |  47 documents             │
  │     │  C:\Users\omen\research\                          │
  │     │  Resuming previous session — 8 exchanges          │  ← if history loaded
  │     │  Last: "What does Chen et al. say about dropout?" │
  │     │  Type /new to start fresh.                        │
  │     └──────────────────────────────────────────────┘
  │
  ├─└► ModelManager.get_embedder()     ← load nomic-embed ONNX (always needed at query time)
  │     ModelManager.get_reranker()     ← load cross-encoder ONNX
  │     ModelManager.get_llm()          ← lazy: load only on first query
  │
  └─└► Enter prompt_toolkit REPL loop
        prompt "motif > "
```

---


### 1.1 Entry Point

```python
# cli.py
def ingest(paths: tuple[str], recursive: bool):
    files = collect_files(paths, recursive, SUPPORTED_EXTENSIONS)
    for filepath in files:
        if not ingestion_tracker.is_indexed(filepath):
            ingest_document(filepath)
        else:
            logger.info(f"Skipping already-indexed: {filepath.name}")
```

### 1.2 Per-Document Flow

```
filepath
  │
  ├─► IngestionTracker.is_indexed(filepath)
  │     → bool (skip if True and hash unchanged)
  │
  ├─► get_parser(filepath.suffix, config) → BasePDFParser | BaseParser
  │     Rules:
  │       .pdf  → is_academic_pdf()? → NougatParser (T3 opt-in)
  │             → needs_ocr()?       → SuryaParser (T3) | PaddleOCRParser (T2)
  │                                  → PyMuPDFParser (T1, text-only PDFs)
  │       .docx → DOCXParser
  │       .md   → MarkdownParser
  │       .png/.jpg/.jpeg/.webp → ImageParser
  │       .mp3/.wav/.m4a/.ogg  → AudioParser
  │
  ├─► parser.extract(filepath) → Extraction
  │     Extraction {
  │       blocks: List[TextBlock]      # Raw text segments with position metadata
  │       image_blocks: List[ImageBlock]  # For PDFs with embedded images
  │       total_pages: int
  │       image_page_count: int
  │     }
  │
  ├─► [Image captioning gate — T3 moondream2, conditional]
  │     if tier == "T3" and moondream_available:
  │       image_ratio = image_page_count / total_pages
  │       if image_ratio >= IMAGE_DENSITY_THRESHOLD (0.3):
  │         ModelManager.load("moondream")
  │         for block in image_blocks:
  │           block.caption = moondream.caption(block.image)
  │         ModelManager.unload("moondream")   ← immediate unload
  │
  ├─► TextNormalizer.normalize(blocks)
  │     → Strips control chars, normalizes whitespace, detects language
  │     → Filters blocks shorter than MIN_CHUNK_TOKENS (64)
  │
  ├─► Chunker.chunk(blocks, config) → List[Chunk]
  │     T1: SentenceChunker   → sentence-boundary split, target 512 tokens, 64 overlap
  │     T2/T3: SemanticChunker → cosine-distance boundary detection, threshold 0.3
  │     Each Chunk { text, char_start, char_end, page_number, section_title, token_count }
  │
  ├─► Deduplicator.filter(chunks, existing_hashes) → List[Chunk]
  │     → SimHash per chunk; drop if Hamming distance < 3 to existing chunk
  │
  ├─► Embedder.encode_batch(chunks) → List[np.ndarray]   shape: (N, embed_dim)
  │     nomic-embed-text-v1.5 ONNX INT8
  │     T1: embed_dim=256 (Matryoshka truncation + re-normalize)
  │     T2/T3: embed_dim=768
  │
  ├─► ChunkStore.insert_batch(chunks) → List[chunk_id: str]
  │     SQLite INSERT with all ChunkMetadata fields
  │
  ├─► VectorStore.upsert_batch(chunk_ids, embeddings, sparse_vectors, metadata)
  │     Qdrant: dense HNSW + sparse vectors (BM25 term weights) per chunk
  │
  ├─► BM25Index.add_batch(chunk_ids, texts)
  │     rank_bm25: rebuild index (pure Python)
  │     tantivy: incremental write (>100K chunks threshold)
  │
  └─► IngestionTracker.update(filepath, content_hash)
      ModelManager.after_ingestion()  ← unload OCR/audio/caption models
```

### 1.3 Modality-Specific Parser Outputs

#### PDF (text)
```
PyMuPDFParser.extract(filepath) → Extraction
  For each page:
    - Text blocks with bbox, font size, font name
    - Table detection (tabula-style bounding boxes)
    - Section heading detection (font size > body_font_size × 1.2)
  Output: TextBlock { text, page_number, section_title, is_table, char_start, char_end }
```

#### PDF (scanned, T2/T3)
```
PaddleOCRParser / SuryaParser.extract(filepath) → Extraction
  → Convert pages to images (300 DPI)
  → OCR each page → text + confidence score
  → Filter: reject blocks with avg confidence < 0.6
  → Stitch into TextBlocks with position metadata
  → is_ocr = True in metadata
```

#### DOCX
```
DOCXParser.extract(filepath) → Extraction
  → Walk paragraphs and tables in document order
  → Heading styles (Heading 1/2/3) → section_title
  → Tables → serialize as markdown table → TextBlock with has_table=True
  → Embedded images → ImageBlock (passed to image captioning gate)
```

#### Markdown
```
MarkdownParser.extract(filepath) → Extraction
  → Parse AST with markdown-it-py
  → Heading nodes → section boundaries
  → Code blocks → separate TextBlock (tagged as code)
  → Tables → markdown table TextBlock
```

#### Image
```
ImageParser.extract(filepath) → Extraction
  → PaddleOCR → extracted text
  → If text empty or sparse (< 20 tokens): ImageBlock for captioning gate
  → Output: TextBlock with is_ocr=True, or ImageBlock
```

#### Audio
```
AudioParser.extract(filepath) → Extraction
  → whisper.cpp transcription → segments with timestamps
  → Each segment: { text, start_time, end_time }
  → Merge segments into ~512-token chunks preserving timestamp boundaries
  → Output: TextBlock with start_time, end_time, page_number=None
```

---

## 2. Query Flow

### 2.1 Entry Point

```python
# REPL loop (cli.py)
while True:
    raw_input = prompt_session.prompt("motif > ")

    if raw_input.startswith("/"):
        handle_slash_command(raw_input, session)
    elif raw_input.strip() in ("exit", "quit", ""):
        session.save()     # persist history to ~/.ragdb/history.json
        break
    else:
        session.pipeline.answer(raw_input, history=session.history)
```

### 2.2 Full Query Pipeline

```
raw_query: str
history: List[Dict]          # last N turns, may be empty
metadata_filter: Optional[QdrantFilter]
  │
  ├─► QueryExpander.should_use_hyde(query, config)
  │     word_count ≤ 7 AND starts with factual marker AND no reasoning marker
  │       → skip HyDE
  │     else (T2/T3 only)
  │       → HYDE_PROMPT → LLMClient.generate() → hypothetical_doc: str
  │         embed_query = hypothetical_doc   (HyDE: embed the fake answer)
  │     T1: always skip HyDE
  │
  ├─► Embedder.encode(embed_query) → query_vector: np.ndarray  shape: (embed_dim,)
  │
  ├─► [Parallel retrieval]
  │     ├─ VectorStore.search_dense(query_vector, top_k, metadata_filter)
  │     │    → List[ScoredPassage]  (Qdrant HNSW)
  │     ├─ VectorStore.search_sparse(query_vector, top_k, metadata_filter)
  │     │    → List[ScoredPassage]  (Qdrant sparse)
  │     └─ BM25Index.search(raw_query, top_k)
  │          → List[ScoredPassage]  (BM25 lexical)
  │
  ├─► rrf_fuse(dense, sparse, bm25, k=60) → top-20: List[ScoredPassage]
  │     RRF score = Σ 1/(k + rank_i) across all lists
  │
  ├─► ChunkStore.fetch_batch(chunk_ids) → chunk texts + full ChunkMetadata
  │
  ├─► CrossEncoder.rerank(raw_query, passages, top_k=3|5) → List[ScoredPassage]
  │     Always runs (never skipped — 85ms is non-negotiable)
  │     Relevance threshold: auto-calibrated (default 0.3)
  │
  ├─► ContextBuilder.build(passages, query, config)
  │     max_tokens=config.llm.max_tokens,
  │     temperature=0.1
  │   ) → Iterator[str]  (token stream)
  │
  └─► Assemble Answer(text, citations, latencies, tier, confidence)
      → CLI: rich.Live → stream tokens as they arrive
      → Citations: format and append after streaming completes
```

---

## 3. Sync Flow (Incremental Update)

```
python cli.py sync ./docs/
  │
  ├─► collect_files(path, recursive=True)
  │     → Set[filepath] = all supported files in directory
  │
  ├─► IngestionTracker.get_all_indexed(root=path)
  │     → Set[filepath] = all files currently in tracker
  │
  ├─► Diff:
  │     new_files     = filesystem - tracker         → ingest each
  │     deleted_files = tracker - filesystem         → delete each
  │     changed_files = filesystem ∩ tracker where hash changed → delete + re-ingest
  │
  ├─► For each deleted_file:
  │     VectorStore.delete_by_source(filepath)
  │     BM25Index.delete_by_source(filepath)         (tantivy: O(1); rank_bm25: rebuild)
  │     ChunkStore.delete_by_source(filepath)
  │     IngestionTracker.remove(filepath)
  │
  └─► For each new/changed file:
      ingest_document(filepath)
```

---

## 4. Error Handling & Fallback Chains

### 4.1 Ingestion Errors

| Error | Behavior |
|---|---|
| Parser fails (corrupt file) | Log error, skip file, continue with remaining |
| OCR confidence < 0.6 | Drop that block; log warning with page number |
| Audio transcription fails | Log error, skip file |
| Embedding fails (OOM) | Reduce batch size by half, retry; if still fails: skip document |
| Qdrant write error | Check disk space; raise with actionable message |

### 4.2 Query Errors

| Error | Fallback |
|---|---|
| HyDE generation fails | Fall back to raw query embedding |
| Qdrant search fails | Fall back to BM25-only retrieval |
| All passages below relevance threshold | Return "No relevant passages found" without invoking LLM |
| LLM generation stalls (>30s) | Timeout; return partial answer with warning |
| LLM OOM | Lower `n_gpu_layers` by 5 and retry once |

### 4.3 Answer Fallback Chain

```python
async def answer_with_fallback(query: str) -> Answer:
    # Attempt 1: Full hybrid + reranking
    answer = await full_pipeline(query)
    if answer.confidence > 0.7:
        return answer

    # Attempt 2: Drop HyDE, retry with raw query embedding
    answer = await full_pipeline(query, force_no_hyde=True)
    if answer.confidence > 0.5:
        return answer

    # Attempt 3: BM25 only (most robust for exact keywords)
    bm25_results = bm25_index.search(query, top_k=10)
    return await generate_from_passages(query, bm25_results)
    # Always returns something; low confidence flagged in Answer
```

---

## 5. Stage Interface Contracts

### Ingestion interfaces

| From | To | Data | Schema |
|---|---|---|---|
| Parser | Chunker | `Extraction` | `blocks: List[TextBlock]`, `image_blocks: List[ImageBlock]` |
| Chunker | Embedder | `List[Chunk]` | `text: str`, `token_count: int`, `char_start/end: int`, `page_number: Optional[int]` |
| Embedder | VectorStore | `List[np.ndarray]` | shape `(N, embed_dim)`, normalized float32 |
| Embedder | VectorStore | `List[Dict]` (sparse) | `{term: weight}` BM25 term weights |
| Chunker | ChunkStore | `List[Chunk]` | Full ChunkMetadata fields |

### Query interfaces

| From | To | Data | Schema |
|---|---|---|---|
| Expander | Embedder | `str` (query or HyDE doc) | Plain text |
| Embedder | VectorStore | `np.ndarray` | shape `(embed_dim,)`, normalized float32 |
| VectorStore | Fusion | `List[ScoredPassage]` | `chunk_id: str`, `score: float` |
| Fusion | ChunkStore | `List[chunk_id]` | UUIDs |
| ChunkStore | CrossEncoder | `List[Passage]` | `text: str`, `metadata: ChunkMetadata` |
| CrossEncoder | ContextBuilder | `List[ScoredPassage]` | `relevance_score: float` |
| ContextBuilder | LLMClient | `str` (context), `str` (query) | Formatted prompt context |
| LLMClient | CLI | `Iterator[str]` | Token stream |
