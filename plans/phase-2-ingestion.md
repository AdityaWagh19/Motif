# Phase 2 — Text Ingestion Pipeline

> **Status:** Not started  
> **Prerequisite:** Phase 1 complete (all storage unit tests pass)  
> **Model downloads required:** `nomic-embed-text-v1.5` ONNX INT8 (~274 MB)  
> **Estimated scope:** 10 files created/modified, ~1,200 lines of implementation

---

## Objective

Build the full ingestion pipeline for PDF and Markdown files. By the end of this
phase, running `/ingest ./docs` produces a populated index visible in `/status`,
and running the same command again produces zero new chunks (deduplication works).

The pipeline flow is:

```
File path
   │
   ▼
Parser (PDF or MD)  →  raw text + metadata per page/section
   │
   ▼
SentenceChunker     →  List[Chunk] with token counts
   │
   ▼
Deduplicator        →  filter near-duplicate chunks (SimHash)
   │
   ▼
Embedder            →  numpy vectors (768-dim, L2-normalised)
   │
   ▼
VectorStore.upsert  →  Qdrant HNSW + sparse vectors
BM25Index.add_batch →  rank_bm25 lexical index
ChunkStore.insert_batch → SQLite chunk text + metadata
IngestionTracker.update → file hash recorded
```

---

## Scope

**In scope:**
- `rag/ingestion/parsers/base.py` — BaseParser ABC
- `rag/ingestion/parsers/pdf.py` — PyMuPDF text extraction
- `rag/ingestion/parsers/markdown.py` — markdown-it-py heading-aware extraction
- `rag/ingestion/chunker.py` — SentenceChunker
- `rag/ingestion/deduplicator.py` — SimHash near-duplicate detection
- `rag/models/embedder.py` — full ONNX implementation (replaces Phase 0 skeleton)
- `rag/retrieval/vector_store.py` — Qdrant local mode wrapper
- `rag/ingestion/__init__.py` — implement `ingest_path()` and `remove_document()`
- Update `rag/commands/ingest.py` — remove `NotImplementedError`, use real pipeline
- Update `rag/commands/status.py` — show real counts from ChunkStore
- `tests/unit/test_parsers.py`
- `tests/unit/test_chunker.py`
- `tests/unit/test_deduplicator.py`
- `tests/unit/test_embedder.py` (marked `@pytest.mark.slow` — needs model file)
- `tests/integration/test_ingestion.py`

**Out of scope:**
- DOCX parser (Phase 4)
- Image parser (Phase 5)
- Audio parser (Phase 5)
- Semantic chunker (Phase 4)
- Query pipeline (Phase 3)

---

## Model Download Requirement

Before running integration tests, the nomic-embed model must be present:

```bash
motif setup --tier T1   # downloads nomic-embed (274 MB) + MiniLM reranker
# OR just the embedder:
python setup_models.py --tier T1
```

Unit tests for parser and chunker do NOT require the model. Only `test_embedder.py`
and `test_ingestion.py` require it.

---

## File Specifications

### `rag/ingestion/parsers/base.py`

**Purpose:** Abstract base class that all parsers implement.

```python
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ParsedPage:
    """One logical unit from a parsed document (one PDF page, one MD section)."""
    text: str
    page: Optional[int] = None          # 1-indexed; None for audio/MD sections
    section: Optional[str] = None       # Nearest heading
    has_table: bool = False
    has_image: bool = False
    is_ocr: bool = False
    start_time: Optional[float] = None  # Audio only
    end_time: Optional[float] = None    # Audio only

class BaseParser(ABC):
    """
    All parsers produce a list of ParsedPage from a file path.
    Parsers do NOT chunk — they return page/section units.
    Chunking is handled by chunker.py.
    """

    SUPPORTED_EXTENSIONS: List[str] = []

    @abstractmethod
    def parse(self, path: Path) -> List[ParsedPage]:
        """
        Parse a file and return one ParsedPage per logical document unit.

        Args:
            path: Absolute path to the file.

        Returns:
            List of ParsedPage objects. Must not return empty text.
            Each ParsedPage.text must be stripped of leading/trailing whitespace.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If the file type is not supported by this parser.
        """
        ...

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

def get_parser(path: Path) -> BaseParser:
    """
    Return the appropriate parser for the given file path.

    Raises:
        ValueError: If no parser supports this file type.
    """
    from rag.ingestion.parsers.pdf import PDFParser
    from rag.ingestion.parsers.markdown import MarkdownParser

    for parser_class in [PDFParser, MarkdownParser]:
        if parser_class.can_parse(path):
            return parser_class()

    raise ValueError(
        f"No parser available for file type '{path.suffix}'. "
        f"Supported: .pdf, .md, .txt"
    )
```

---

### `rag/ingestion/parsers/pdf.py`

**Backend:** `pymupdf` (import as `fitz`)

**Strategy:**
1. Open document with `fitz.open(path)`
2. For each page: extract text with `page.get_text("text")`
3. If page text is empty (scanned PDF): set `is_ocr = True` and leave text empty
   (OCR is Phase 5). Log a warning.
4. Detect tables: `len(page.find_tables().tables) > 0`
5. Detect images: `len(page.get_images()) > 0`
6. Detect nearest section title: scan text for lines that are ALL CAPS or match
   `^\d+\.\s+[A-Z]` (numbered heading pattern). Take the last matching line above
   the current position.

```python
SUPPORTED_EXTENSIONS = [".pdf"]

def parse(self, path: Path) -> List[ParsedPage]:
    pages = []
    doc = fitz.open(str(path))
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        is_ocr = False
        if not text:
            # Scanned page — no text layer
            is_ocr = True
            # Phase 5 will fill this in with PaddleOCR/Surya
            # For now: skip empty pages (they produce no chunks)
            continue

        pages.append(ParsedPage(
            text=text,
            page=page_num,
            section=_detect_section(text),
            has_table=len(page.find_tables().tables) > 0,
            has_image=len(page.get_images()) > 0,
            is_ocr=is_ocr,
        ))
    doc.close()
    return pages

def _detect_section(text: str) -> Optional[str]:
    """
    Heuristic: return the first line that looks like a section heading.
    Criteria: line <= 80 chars, not ending in punctuation, contains >= 2 words.
    """
    for line in text.splitlines():
        line = line.strip()
        if 5 <= len(line) <= 80 and not line.endswith((".", ",", ":", ";")):
            words = line.split()
            if 2 <= len(words) <= 10:
                return line
    return None
```

---

### `rag/ingestion/parsers/markdown.py`

**Backend:** `markdown_it` from `markdown-it-py`

**Strategy:**
1. Parse the file with `MarkdownIt().parse(content)`
2. Walk tokens to extract heading text and paragraph text
3. Each H1/H2/H3 starts a new section
4. Accumulate paragraph text within each section
5. Return one `ParsedPage` per section (or the whole file if no headings)

```python
SUPPORTED_EXTENSIONS = [".md", ".txt", ".markdown"]

def parse(self, path: Path) -> List[ParsedPage]:
    content = path.read_text(encoding="utf-8", errors="replace")
    md = MarkdownIt()
    tokens = md.parse(content)

    sections: List[ParsedPage] = []
    current_heading: Optional[str] = None
    current_text_parts: List[str] = []

    for token in tokens:
        if token.type == "heading_open":
            # Flush previous section
            if current_text_parts:
                sections.append(ParsedPage(
                    text=" ".join(current_text_parts).strip(),
                    section=current_heading,
                ))
                current_text_parts = []
        elif token.type == "inline" and token.content:
            parent = getattr(token, "parent", None)
            # If previous token was a heading, this is the heading text
            # Detect by checking if the last heading_open is still "open"
            if _is_heading_inline(tokens, token):
                current_heading = token.content.strip()
            else:
                current_text_parts.append(token.content)

    # Flush final section
    if current_text_parts:
        sections.append(ParsedPage(
            text=" ".join(current_text_parts).strip(),
            section=current_heading,
        ))

    # If nothing extracted, return raw content as one page
    if not sections:
        sections = [ParsedPage(text=content.strip())]

    return [s for s in sections if s.text]

def _is_heading_inline(tokens, current_token) -> bool:
    """Return True if the current inline token is a heading title."""
    for i, tok in enumerate(tokens):
        if tok is current_token:
            return i > 0 and tokens[i - 1].type == "heading_open"
    return False
```

---

### `rag/ingestion/chunker.py`

**Purpose:** Split `ParsedPage` objects into fixed-size `Chunk` objects with overlap.

**Phase 2 implementation:** `SentenceChunker` — splits on sentence boundaries
(`.`, `!`, `?` followed by space or newline) with a target token count and overlap.

**Phase 4 addition:** `SemanticChunker` — uses embedding cosine distance.

**Token counting:** Use `len(text.split())` as an approximation. This is ≈75% of
true BPE token count but is zero-dependency and fast. Add a 33% safety margin to
the target (512 tokens target → allow up to 680 words before splitting).

```python
WORDS_PER_TOKEN_RATIO = 0.75   # words ≈ 0.75 × tokens
OVERLAP_WORDS = int(64 / WORDS_PER_TOKEN_RATIO)  # ≈ 85 words

@dataclass
class ChunkerConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64

class SentenceChunker:
    """
    Splits text on sentence boundaries. Target: ~512 tokens per chunk.
    Overlap: ~64 tokens from the end of the previous chunk.
    """

    def __init__(self, config: ChunkerConfig = ChunkerConfig()) -> None:
        self._target_words = int(config.target_tokens / WORDS_PER_TOKEN_RATIO)
        self._overlap_words = int(config.overlap_tokens / WORDS_PER_TOKEN_RATIO)

    def chunk(
        self,
        page: "ParsedPage",
        source: str,
        filename: str,
        source_type: str,
    ) -> List[Chunk]:
        """
        Chunk a ParsedPage into a list of Chunk objects.

        Algorithm:
        1. Split text into sentences using regex: (?<=[.!?])\s+
        2. Accumulate sentences until word count exceeds target_words
        3. Create a chunk, then start next chunk with last overlap_words words
           from the previous chunk (overlap)
        4. Assign char_start/char_end relative to page text
        5. Generate UUID for each chunk ID
        6. Set token_count = len(text.split())  [word approximation]
        """
        sentences = re.split(r'(?<=[.!?])\s+', page.text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_word_count = 0
        overlap_buffer: List[str] = []  # words for next chunk's lead-in

        def make_chunk(sentences_: List[str]) -> Chunk:
            text_ = " ".join(sentences_)
            return Chunk(
                id=str(uuid.uuid4()),
                text=text_,
                source=source,
                filename=filename,
                source_type=source_type,
                page=page.page,
                section=page.section,
                has_table=page.has_table,
                has_image=page.has_image,
                is_ocr=page.is_ocr,
                token_count=len(text_.split()),
                indexed_at=datetime.utcnow().isoformat() + "Z",
            )

        for sentence in sentences:
            words = sentence.split()
            if current_word_count + len(words) > self._target_words and current_sentences:
                chunks.append(make_chunk(current_sentences))
                # Build overlap: take last overlap_words words from current chunk
                overlap_text = " ".join(current_sentences)
                overlap_words = overlap_text.split()[-self._overlap_words:]
                current_sentences = [" ".join(overlap_words), sentence] if overlap_words else [sentence]
                current_word_count = sum(len(s.split()) for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_word_count += len(words)

        if current_sentences:
            chunks.append(make_chunk(current_sentences))

        return chunks

    def chunk_pages(
        self,
        pages: List["ParsedPage"],
        source: str,
        filename: str,
        source_type: str,
    ) -> List[Chunk]:
        """Chunk all pages from a document. Returns flat list of Chunk objects."""
        all_chunks = []
        for page in pages:
            all_chunks.extend(self.chunk(page, source, filename, source_type))
        return all_chunks
```

---

### `rag/ingestion/deduplicator.py`

**Purpose:** Detect near-duplicate chunks using SimHash. Prevents nearly-identical
content (e.g., repeated boilerplate headers) from appearing multiple times in the
index and inflating retrieval scores.

**Backend:** `simhash` library (`pip install simhash`)

**Threshold:** Hamming distance ≤ 3 (out of 64 bits) → near-duplicate

```python
from simhash import Simhash

DUPLICATE_HAMMING_THRESHOLD = 3

class Deduplicator:
    """
    SimHash-based near-duplicate detector.
    Maintains an in-memory set of seen hashes during an ingestion run.
    Not persisted between runs (ChunkStore content_hash handles cross-run dedup).
    """

    def __init__(self, threshold: int = DUPLICATE_HAMMING_THRESHOLD) -> None:
        self._threshold = threshold
        self._seen: List[Tuple[Simhash, str]] = []  # (hash, chunk_id)

    def _compute_hash(self, text: str) -> Simhash:
        """Tokenise to trigrams for SimHash computation."""
        tokens = [text[i:i+3] for i in range(len(text) - 2)]  # character trigrams
        return Simhash(tokens)

    def is_duplicate(self, chunk: Chunk) -> bool:
        """
        Return True if this chunk is a near-duplicate of a previously seen chunk.
        Side effect: if not duplicate, adds this chunk's hash to seen set.
        """
        h = self._compute_hash(chunk.text)
        for seen_hash, _ in self._seen:
            if h.distance(seen_hash) <= self._threshold:
                return True
        self._seen.append((h, chunk.id))
        return False

    def filter(self, chunks: List[Chunk]) -> List[Chunk]:
        """Filter a list of chunks, removing near-duplicates. Keeps first occurrence."""
        return [c for c in chunks if not self.is_duplicate(c)]

    def reset(self) -> None:
        """Clear the seen set. Call between documents to avoid cross-document dedup."""
        self._seen = []
```

---

### `rag/models/embedder.py` (full implementation)

**Replaces the Phase 0 skeleton.**

**Model:** `nomic-embed-text-v1.5` ONNX INT8
**Model directory structure:**
```
models/nomic-embed-text-v1.5/
    onnx/model_quantized.onnx
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    config.json
```

**Dependencies:** `onnxruntime`, `tokenizers`

```python
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

EMBEDDING_DIM = 768
MAX_SEQ_LEN = 8192  # nomic-embed-text supports up to 8192 tokens

class Embedder:

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer: Optional[Tokenizer] = None

    def _load(self) -> None:
        onnx_path = self._model_dir / "onnx" / "model_quantized.onnx"
        tok_path = self._model_dir / "tokenizer.json"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

        providers = ["CPUExecutionProvider"]  # GPU provider added by Phase 3 if needed
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        self._tokenizer.enable_padding(pad_id=1, pad_token="[PAD]", length=None)

    def _mean_pool_and_normalize(
        self,
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Mean pool token embeddings (masked) then L2-normalize.
        token_embeddings: (batch, seq_len, 768)
        attention_mask:   (batch, seq_len)
        Returns: (batch, 768) float32
        """
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = (token_embeddings * mask_expanded).sum(axis=1)
        count = mask_expanded.sum(axis=1).clip(min=1e-9)
        mean_embeddings = sum_embeddings / count
        # L2 normalise
        norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True).clip(min=1e-9)
        return (mean_embeddings / norms).astype(np.float32)

    def _tokenize(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize a batch of texts. Returns (input_ids, attention_mask)."""
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        return input_ids, attention_mask

    def encode(self, text: str, prefix: str = "search_query: ") -> np.ndarray:
        """Encode a single text. Returns (768,) float32 array."""
        if self._session is None:
            raise RuntimeError("Embedder not loaded. Call _load() first.")
        return self.encode_batch([text], prefix=prefix)[0]

    def encode_batch(
        self,
        texts: List[str],
        prefix: str = "search_document: ",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a list of texts in mini-batches.
        Returns (N, 768) float32 array, each row L2-normalised.
        """
        if self._session is None:
            raise RuntimeError("Embedder not loaded. Call _load() first.")

        prefixed = [f"{prefix}{t}" for t in texts]
        all_embeddings = []

        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            input_ids, attention_mask = self._tokenize(batch)

            outputs = self._session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            token_embeddings = outputs[0]  # (batch, seq_len, 768)
            embeddings = self._mean_pool_and_normalize(token_embeddings, attention_mask)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def is_loaded(self) -> bool:
        return self._session is not None

    def unload(self) -> None:
        self._session = None
        self._tokenizer = None
```

---

### `rag/retrieval/vector_store.py`

**Backend:** `qdrant-client` in local mode (no server process)

**Collection configuration:**
- HNSW dense vectors: 768-dim, cosine distance
- Sparse vectors: `BAAI/bge-m3` SPLADE format — but for Phase 2, we use nomic's
  sparse output if available, else dense-only retrieval
- `on_disk=True` — vectors stored on disk, not RAM, for T1 memory budget
- `hnsw_config`: `m=16`, `ef_construct=100` — quality vs. speed tradeoff

**Phase 2 simplification:** Use dense-only retrieval. Sparse retrieval (for full
hybrid search) is introduced in Phase 3 when the full pipeline is in place.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, Range,
)

COLLECTION_NAME = "motif_chunks"
VECTOR_SIZE = 768

class VectorStore:

    def __init__(self, config: RAGConfig) -> None:
        db_path = config.db_root / "qdrant"
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(db_path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                hnsw_config={"m": 16, "ef_construct": 100, "on_disk": True},
            )

    def upsert(self, chunk_id: str, vector: np.ndarray, payload: dict) -> None:
        """Insert or update a single vector with its payload."""
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=_str_to_uuid_int(chunk_id),
                vector=vector.tolist(),
                payload={**payload, "chunk_id": chunk_id},
            )],
        )

    def upsert_batch(
        self,
        chunk_ids: List[str],
        vectors: np.ndarray,
        payloads: List[dict],
    ) -> None:
        """Batch upsert. vectors shape: (N, 768)."""
        points = [
            PointStruct(
                id=_str_to_uuid_int(cid),
                vector=vec.tolist(),
                payload={**payload, "chunk_id": cid},
            )
            for cid, vec, payload in zip(chunk_ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filter_: Optional[dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Dense HNSW search. Returns [(chunk_id, score)] sorted by score descending.
        filter_ dict keys: "source" (str), "source_type" (str), "page_min" (int),
        "page_max" (int). All are optional.
        """
        qdrant_filter = _build_filter(filter_) if filter_ else None
        results = self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [(r.payload["chunk_id"], r.score) for r in results]

    def delete_by_source(self, source: str) -> int:
        """
        Delete all vectors whose payload.source matches the given path.
        Returns approximate count (Qdrant does not guarantee exact count on delete).
        """
        from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue
        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ),
        )
        return 0  # Qdrant delete does not return count; caller uses ChunkStore count

    def count(self) -> int:
        return self._client.count(collection_name=COLLECTION_NAME).count

def _str_to_uuid_int(s: str) -> int:
    """Convert a UUID string to an integer for Qdrant point ID."""
    import uuid as uuid_module
    return uuid_module.UUID(s).int

def _build_filter(filter_dict: dict) -> Filter:
    """Build a Qdrant Filter from a plain dict of metadata constraints."""
    conditions = []
    if "source" in filter_dict:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=filter_dict["source"])))
    if "source_type" in filter_dict:
        conditions.append(FieldCondition(key="source_type", match=MatchValue(value=filter_dict["source_type"])))
    if "page_min" in filter_dict or "page_max" in filter_dict:
        conditions.append(FieldCondition(
            key="page",
            range=Range(
                gte=filter_dict.get("page_min"),
                lte=filter_dict.get("page_max"),
            ),
        ))
    return Filter(must=conditions) if conditions else None
```

---

### `rag/ingestion/__init__.py` (full implementation)

Replaces the Phase 0 stub. Implements `ingest_path()` and `remove_document()`.
`sync_directory()` remains a stub until Phase 4.

```
ingest_path(path, config, recursive, console) -> IngestResult:

    1. Validate path exists (caller already does this — add assert)
    2. Collect files:
       - If file: files = [path]
       - If directory + not recursive: files = list(path.iterdir() where is_file())
       - If directory + recursive: files = list(path.rglob("*") where is_file())
    3. Filter to supported extensions: .pdf, .md, .txt, .markdown
    4. Initialize: tracker = IngestionTracker(config)
                   chunk_store = ChunkStore(config)
                   bm25 = BM25Index(config)
                   vector_store = VectorStore(config)
                   embedder = model_manager.get_embedder(config)
                   chunker = SentenceChunker(ChunkerConfig(...from config...))
                   deduplicator = Deduplicator()
    5. For each file:
       a. Compute hash = compute_file_hash(file)
       b. If tracker.is_indexed(file) and tracker.get_hash(file) == hash:
              skipped += 1; continue  [hash unchanged — skip]
       c. If tracker.is_indexed(file) and hash differs:
              remove_document(file, config)  [re-index]
       d. parser = get_parser(file)
       e. pages = parser.parse(file)  [may raise — catch, log, record error]
       f. chunks = chunker.chunk_pages(pages, str(file.resolve()), file.name, source_type)
       g. chunks = deduplicator.filter(chunks)  [drop near-dups]
       h. deduplicator.reset()  [reset between documents]
       i. If no chunks: log warning, continue
       j. vectors = embedder.encode_batch([c.text for c in chunks],
                                          prefix="search_document: ")
       k. chunk_store.insert_batch(chunks)
       l. bm25.add_batch(chunks)
       m. payloads = [chunk_to_payload(c) for c in chunks]
          vector_store.upsert_batch([c.id for c in chunks], vectors, payloads)
       n. tracker.update(file, hash, len(chunks))
       o. files_processed += 1; chunks_added += len(chunks)
       p. If console: print progress line
    6. bm25.save()
    7. model_manager.after_ingestion(config)  [unload embedder on T1]
    8. Return IngestResult(files_processed, chunks_added, skipped, errors)

def chunk_to_payload(chunk: Chunk) -> dict:
    """Extract Qdrant payload fields from a Chunk (no text — text is in ChunkStore)."""
    return {
        "source": chunk.source,
        "filename": chunk.filename,
        "source_type": chunk.source_type,
        "page": chunk.page,
        "section": chunk.section,
        "has_table": chunk.has_table,
        "has_image": chunk.has_image,
    }

remove_document(path, config) -> int:
    1. source = str(path.resolve())
    2. chunk_store = ChunkStore(config)
    3. n = chunk_store.delete_by_source(source)
    4. vector_store.delete_by_source(source)
    5. bm25.delete_by_source(source, chunk_ids)  [need chunk_ids — get from store first]

    Corrected order:
    a. chunk_store = ChunkStore(config); bm25 = BM25Index(config); vector_store = VectorStore(config)
    b. chunk_ids_to_remove = [c.id for c in chunk_store.fetch_by_source(source)]
       [Add fetch_by_source() to ChunkStore: SELECT id WHERE source = ?]
    c. n = chunk_store.delete_by_source(source)
    d. bm25.delete_by_source(source, chunk_ids_to_remove); bm25.save()
    e. vector_store.delete_by_source(source)
    f. tracker = IngestionTracker(config); tracker.remove(path)
    g. Return n
```

**Additional method needed in ChunkStore:**
```
fetch_by_source(self, source: str) -> List[Chunk]
    SELECT * FROM chunks WHERE source = ?
```
Add this method to ChunkStore in this phase.

---

### Update `rag/commands/ingest.py`

Remove the `ImportError` branch. The real implementation is now in `rag.ingestion`.
Keep the argument parsing. Add a Rich progress callback.

```python
from rag.ingestion import ingest_path

result = ingest_path(target, config=config, recursive=parsed.recursive, console=console)
console.print(
    f"[green]Ingestion complete.[/green] "
    f"Files: {result.files_processed}  "
    f"Chunks added: {result.chunks_added}  "
    f"Skipped (unchanged): {result.files_skipped}"
)
if result.errors:
    for err in result.errors:
        console.print(f"  [red]error:[/red] {err}")
```

### Update `rag/commands/status.py`

Use real ChunkStore and BM25Index counts:

```python
from rag.storage.chunk_store import ChunkStore
from rag.retrieval.bm25_index import BM25Index

store = ChunkStore(config)
bm25 = BM25Index(config)
chunk_count = store.count()
doc_count = store.count_documents()
bm25_count = bm25.count()
```

---

## Test Specifications

### `tests/unit/test_parsers.py`

Uses `sample_md` fixture from conftest. Does NOT need model.

```
test_markdown_parser_extracts_sections:
    pages = MarkdownParser().parse(sample_md)
    assert len(pages) >= 2
    headings = [p.section for p in pages if p.section]
    assert "Introduction" in headings or any("Intro" in h for h in headings)

test_markdown_parser_non_empty_text:
    pages = MarkdownParser().parse(sample_md)
    for p in pages:
        assert p.text.strip()

test_pdf_parser_skips_empty_pages:
    # Create a mock: patch fitz.open to return pages with empty text
    # Verify those pages are skipped (not returned)

test_get_parser_pdf:
    parser = get_parser(Path("document.pdf"))
    assert isinstance(parser, PDFParser)

test_get_parser_md:
    parser = get_parser(Path("notes.md"))
    assert isinstance(parser, MarkdownParser)

test_get_parser_unsupported:
    with pytest.raises(ValueError):
        get_parser(Path("video.mp4"))
```

### `tests/unit/test_chunker.py`

```
test_sentence_chunker_basic:
    page = ParsedPage(text="First sentence. Second sentence. Third sentence.", page=1)
    chunks = chunker.chunk(page, "/test.pdf", "test.pdf", "pdf")
    assert all(c.source == "/test.pdf" for c in chunks)
    assert all(c.page == 1 for c in chunks)

test_sentence_chunker_long_text_splits:
    # 600-word text should produce at least 2 chunks
    long_text = " ".join(["This is sentence number {}.".format(i) for i in range(100)])
    page = ParsedPage(text=long_text)
    chunks = chunker.chunk(page, "/doc.md", "doc.md", "md")
    assert len(chunks) >= 2

test_sentence_chunker_overlap:
    long_text = " ".join([f"Word{i}" for i in range(200)])
    page = ParsedPage(text=long_text)
    chunks = chunker.chunk(page, "/doc.md", "doc.md", "md")
    if len(chunks) >= 2:
        # Last words of chunk[0] should appear at start of chunk[1]
        words_end_c0 = chunks[0].text.split()[-10:]
        words_start_c1 = chunks[1].text.split()[:10]
        assert any(w in words_start_c1 for w in words_end_c0)

test_chunk_has_uuid:
    import uuid
    page = ParsedPage(text="Test content for UUID check.")
    chunks = chunker.chunk(page, "/test.md", "test.md", "md")
    for c in chunks:
        uuid.UUID(c.id)  # raises ValueError if not valid UUID

test_chunk_pages_multiple_pages:
    pages = [ParsedPage(text=f"Page {i} content with enough words." * 5, page=i) for i in range(3)]
    chunks = chunker.chunk_pages(pages, "/doc.pdf", "doc.pdf", "pdf")
    assert len(chunks) >= 3
```

### `tests/unit/test_deduplicator.py`

```
test_no_duplicate:
    chunks = [Chunk(id="1", text="The quick brown fox", ...),
              Chunk(id="2", text="A completely different text about databases", ...)]
    result = dedup.filter(chunks)
    assert len(result) == 2

test_near_duplicate_removed:
    text = "The quick brown fox jumped over the lazy dog near the river"
    c1 = Chunk(id="1", text=text, ...)
    c2 = Chunk(id="2", text=text + " ", ...)  # nearly identical
    result = dedup.filter([c1, c2])
    assert len(result) == 1
    assert result[0].id == "1"  # first occurrence kept

test_reset_clears_state:
    text = "Identical text appears twice"
    c1 = Chunk(id="1", text=text, ...)
    c2 = Chunk(id="2", text=text, ...)
    dedup.filter([c1])
    dedup.reset()
    result = dedup.filter([c2])
    assert len(result) == 1  # c2 is accepted after reset
```

### `tests/unit/test_embedder.py`

```python
# Marked @pytest.mark.slow — requires model file at models/nomic-embed-text-v1.5/
import pytest

@pytest.mark.slow
def test_encode_returns_correct_shape(embedder):
    vec = embedder.encode("test sentence")
    assert vec.shape == (768,)
    assert vec.dtype == np.float32

@pytest.mark.slow
def test_encode_is_normalized(embedder):
    vec = embedder.encode("test sentence")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5  # L2 norm ≈ 1.0

@pytest.mark.slow
def test_encode_batch(embedder):
    texts = ["first text", "second text", "third text"]
    vecs = embedder.encode_batch(texts)
    assert vecs.shape == (3, 768)

@pytest.mark.slow
def test_similar_texts_closer_than_dissimilar(embedder):
    v_cat1 = embedder.encode("cats are domestic animals")
    v_cat2 = embedder.encode("felines are pets kept at home")
    v_code = embedder.encode("for loop in Python programming language")
    sim_cats = float(v_cat1 @ v_cat2)
    sim_diff = float(v_cat1 @ v_code)
    assert sim_cats > sim_diff
```

### `tests/integration/test_ingestion.py`

```python
@pytest.mark.slow
def test_ingest_markdown_end_to_end(minimal_config, sample_md):
    result = ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    assert result.files_processed == 1
    assert result.chunks_added >= 1
    assert result.files_skipped == 0

    store = ChunkStore(minimal_config)
    assert store.count() >= 1
    assert store.count_documents() == 1

    bm25 = BM25Index(minimal_config)
    assert bm25.count() >= 1

@pytest.mark.slow
def test_ingest_twice_skips_unchanged(minimal_config, sample_md):
    ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    result2 = ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    assert result2.files_skipped == 1
    assert result2.chunks_added == 0

@pytest.mark.slow
def test_remove_document(minimal_config, sample_md):
    ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    store = ChunkStore(minimal_config)
    assert store.count() >= 1
    n = remove_document(sample_md, config=minimal_config)
    assert n >= 1
    assert store.count() == 0
```

---

## Validation Checklist

```bash
# 1. Imports clean
python -c "from rag.ingestion.parsers.base import BaseParser, get_parser; print('OK')"
python -c "from rag.ingestion.chunker import SentenceChunker; print('OK')"
python -c "from rag.models.embedder import Embedder; print('OK')"
python -c "from rag.retrieval.vector_store import VectorStore; print('OK')"

# 2. Unit tests (no model needed)
pytest tests/unit/test_parsers.py tests/unit/test_chunker.py tests/unit/test_deduplicator.py -v

# 3. Slow tests (need nomic-embed model)
pytest tests/unit/test_embedder.py tests/integration/test_ingestion.py -v -m slow

# 4. Functional test — run REPL and verify /ingest and /status work
#    (manual verification)
# motif
# /ingest ./project-context -r
# /status
# Expected: Documents > 0, Chunks > 0, BM25 count > 0

# 5. Dedup test
# Run /ingest again on same dir → "Skipped (unchanged): N, Chunks added: 0"

# 6. Remove test
# /remove ./project-context/context.md
# /status → count reduced
```

---

## Post-Phase Documentation Updates

**`project-context/progress.md`:**
- Mark Phase 1 tasks for parsers, chunker, embedder, vector_store, ingestion pipeline ✅
- `/ingest` and `/status` working → mark those acceptance criteria ✅
- Phase 1 status: still 🔄 (query pipeline not yet done)

**`project-context/tests.md`:**
- Mark ING-01 through ING-12, EMB-01 through EMB-04 as passing

**Add metrics snapshot:**
- Ingestion throughput: approximately N chunks/second on T1 CPU
