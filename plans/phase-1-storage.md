# Phase 1 — Storage Foundation

> **Status:** Not started  
> **Prerequisite:** Phase 0 complete (motif REPL launches, all stubs in place)  
> **Model downloads required:** None  
> **Estimated scope:** 4 files created, 3 test files, ~600 lines of implementation

---

## Objective

Build the complete storage layer — SQLite chunk store, file hash tracker, BM25
index, and ModelManager — with no model dependencies. Every module in this phase
is pure Python or SQLite. All tests run offline without any downloaded models.

This phase establishes the data layer that every later phase depends on.
Nothing in Phase 2 can be implemented until Phase 1 passes all acceptance criteria.

---

## Scope

**In scope:**
- `rag/storage/chunk_store.py` — SQLite CRUD for `Chunk` objects
- `rag/storage/ingestion_tracker.py` — SHA-256 file hash tracking
- `rag/retrieval/bm25_index.py` — rank_bm25 wrapper with persistence
- `rag/models/model_manager.py` — lazy-load singleton for Embedder/Reranker/LLM
- `rag/storage/__init__.py` — package init
- `rag/retrieval/__init__.py` — package init
- `rag/reranking/__init__.py` — package init
- `rag/generation/__init__.py` — package init
- `tests/unit/test_chunk_store.py`
- `tests/unit/test_tracker.py`
- `tests/unit/test_bm25.py`

**Out of scope:**
- Qdrant vector store (Phase 2)
- Embedder implementation (Phase 2)
- Any parser, chunker, or LLM code (Phase 2+)

---

## File Specifications

### `rag/storage/__init__.py`
Empty package init.

---

### `rag/retrieval/__init__.py`
Empty package init.

---

### `rag/reranking/__init__.py`
Empty package init.

---

### `rag/generation/__init__.py`
Empty package init.

---

### `rag/storage/chunk_store.py`

**Purpose:** SQLite-backed store for all `Chunk` objects. This is the authoritative
record of every indexed chunk — Qdrant holds the vectors, chunk_store holds the text
and full metadata.

**Database location:** `config.db_root / "chunks.db"`

**SQLite settings:**
- WAL mode (`PRAGMA journal_mode=WAL`) — allows concurrent reads during ingestion
- `PRAGMA synchronous=NORMAL` — safe with WAL, faster than FULL
- `PRAGMA foreign_keys=ON`

**Table schema:**

```sql
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    source       TEXT NOT NULL,
    filename     TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    char_start   INTEGER NOT NULL DEFAULT 0,
    char_end     INTEGER NOT NULL DEFAULT 0,
    page         INTEGER,
    section      TEXT,
    start_time   REAL,
    end_time     REAL,
    has_table    INTEGER NOT NULL DEFAULT 0,
    has_image    INTEGER NOT NULL DEFAULT 0,
    is_ocr       INTEGER NOT NULL DEFAULT 0,
    language     TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    token_count  INTEGER NOT NULL DEFAULT 0,
    indexed_at   TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
```

**Class: `ChunkStore`**

```
__init__(self, config: RAGConfig) -> None
    - Open connection to config.db_root / "chunks.db"
    - Create parent directories if missing
    - Apply PRAGMA settings
    - Run CREATE TABLE IF NOT EXISTS

insert(self, chunk: Chunk) -> None
    - INSERT OR REPLACE INTO chunks (all fields)
    - Convert bool fields to 0/1 integers for SQLite

insert_batch(self, chunks: List[Chunk]) -> None
    - Single transaction: BEGIN → N inserts → COMMIT
    - Use executemany() for efficiency
    - Silently skip empty list

fetch(self, chunk_id: str) -> Optional[Chunk]
    - SELECT all columns WHERE id = ?
    - Return None if not found
    - Convert 0/1 integers back to bool

fetch_batch(self, chunk_ids: List[str]) -> List[Chunk]
    - SELECT all WHERE id IN (?, ?, ...)
    - Returns only found chunks (order not guaranteed)
    - Return [] for empty input

delete_by_source(self, source: str) -> int
    - DELETE WHERE source = ?
    - Return rowcount (number of chunks deleted)

count(self) -> int
    - SELECT COUNT(*) FROM chunks

count_documents(self) -> int
    - SELECT COUNT(DISTINCT source) FROM chunks

list_sources(self) -> List[str]
    - SELECT DISTINCT source FROM chunks ORDER BY source
    - Used by /status and /sync

close(self) -> None
    - Close the SQLite connection
```

**Error handling:**
- All database errors propagate as `sqlite3.Error` — do not swallow them
- Log the operation and chunk_id at DEBUG level before each query
- `insert_batch` rolls back the entire transaction if any row fails

---

### `rag/storage/ingestion_tracker.py`

**Purpose:** Track which files have been ingested and with which content hash.
Used during `/ingest` to skip unchanged files and during `/sync` to detect
deleted, added, or changed files.

**Database location:** `config.db_root / "ingestion_tracker.db"`

**Table schema:**

```sql
CREATE TABLE IF NOT EXISTS files (
    filepath     TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at   TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0
);
```

**Class: `IngestionTracker`**

```
__init__(self, config: RAGConfig) -> None
    - Open connection to config.db_root / "ingestion_tracker.db"
    - Apply PRAGMA WAL, NORMAL sync
    - CREATE TABLE IF NOT EXISTS

is_indexed(self, path: Path) -> bool
    - SELECT 1 WHERE filepath = str(path.resolve())

get_hash(self, path: Path) -> Optional[str]
    - SELECT content_hash WHERE filepath = ?
    - Return None if not tracked

update(self, path: Path, content_hash: str, chunk_count: int) -> None
    - INSERT OR REPLACE INTO files VALUES (...)
    - indexed_at = datetime.utcnow().isoformat() + "Z"
    - filepath stored as str(path.resolve()) — always absolute

remove(self, path: Path) -> None
    - DELETE WHERE filepath = str(path.resolve())

list_all(self) -> List[dict]
    - SELECT * FROM files
    - Return list of dicts with keys: filepath, content_hash, indexed_at, chunk_count
    - Used by /sync to compute diff against filesystem
```

**Content hash computation (helper function, not a method):**

```python
def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 of file contents. Returns hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

Export `compute_file_hash` at module level — it will be imported by the ingestion
pipeline in Phase 2.

---

### `rag/retrieval/bm25_index.py`

**Purpose:** BM25 lexical index over chunk text. Complements dense vector search.
Persisted to disk as a pickle file so it survives process restarts.

**Backend:** `rank_bm25.BM25Okapi`

**Persistence location:** `config.db_root / "bm25" / "index.pkl"`

**Pickle format:**
```python
{
    "corpus_tokens": List[List[str]],   # tokenized text per chunk
    "chunk_ids":     List[str],          # parallel list of chunk IDs
    "version":       int,                # schema version (currently 1)
}
```

**Tokenization:** `text.lower().split()` — no stemming, no stopword removal.
Consistent, reproducible, and fast enough for the target corpus size.

**Class: `BM25Index`**

```
__init__(self, config: RAGConfig) -> None
    - Set self._index_path = config.db_root / "bm25" / "index.pkl"
    - self._corpus_tokens: List[List[str]] = []
    - self._chunk_ids: List[str] = []
    - self._bm25: Optional[BM25Okapi] = None
    - self._dirty = False  # True when corpus changed but BM25 not rebuilt
    - Call self._load() if index file exists

_load(self) -> None
    - Unpickle index file
    - Populate self._corpus_tokens and self._chunk_ids
    - Call self._rebuild_bm25()
    - If unpickling fails (corrupt file): log warning, start fresh

_rebuild_bm25(self) -> None
    - If corpus is empty: self._bm25 = None, return
    - self._bm25 = BM25Okapi(self._corpus_tokens)
    - self._dirty = False

save(self) -> None
    - If not self._dirty: return (no-op)
    - Create parent directory if missing
    - Write to a temp file first, then atomic rename to _index_path
    - This prevents corruption if the process dies mid-write

add(self, chunk: Chunk) -> None
    - If chunk.id already in self._chunk_ids: call delete(chunk.id) first
    - Tokenize: tokens = chunk.text.lower().split()
    - Append tokens to self._corpus_tokens
    - Append chunk.id to self._chunk_ids
    - self._dirty = True
    - Call self._rebuild_bm25()  [small corpus — rebuild is fast]

add_batch(self, chunks: List[Chunk]) -> None
    - For each chunk: handle duplicates (delete first if exists)
    - Bulk append all tokens and ids
    - Single call to self._rebuild_bm25() at the end
    - Call self.save()

search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]
    - If self._bm25 is None or corpus empty: return []
    - query_tokens = query.lower().split()
    - scores = self._bm25.get_scores(query_tokens)  [numpy array]
    - Get indices of top_k highest scores
    - Return [(self._chunk_ids[i], float(scores[i])) for i in top_k_indices]
    - Filter out scores == 0.0 (no match)

delete(self, chunk_id: str) -> bool
    - Find index of chunk_id in self._chunk_ids
    - If not found: return False
    - Remove from both lists at that index
    - self._dirty = True
    - self._rebuild_bm25()
    - Return True

rebuild(self) -> None
    - Force full rebuild of BM25Okapi from current corpus
    - Save to disk
    - Use case: after bulk deletion via delete_by_source

delete_by_source(self, source: str, chunk_ids: List[str]) -> int
    - Remove all chunk_ids from the index
    - Return count removed
    - Single rebuild after all deletions

count(self) -> int
    - Return len(self._chunk_ids)
```

---

### `rag/models/model_manager.py`

**Purpose:** Singleton that lazy-loads and manages the lifecycle of all model
instances. No module instantiates Embedder, Reranker, or LLMClient directly — all
access goes through ModelManager.

This module is implemented in Phase 1 because the storage layer needs to know
the interface. The models themselves (Embedder, Reranker, LLMClient) are
implemented in Phases 2–3.

**Singleton pattern:** Thread-safe via a module-level instance, not via `__new__`.
Since Motif is single-threaded (single-user CLI), a simple module-level singleton
is sufficient.

**Class: `ModelManager`**

```
__init__(self) -> None
    - self._embedder: Optional["Embedder"] = None
    - self._reranker: Optional["Reranker"] = None
    - self._llm: Optional["LLMClient"] = None
    - self._embedder_config_hash: Optional[str] = None  # detect config changes

get_embedder(self, config: RAGConfig) -> "Embedder"
    - Lazy import: from rag.models.embedder import Embedder
    - If self._embedder is None:
        model_path = Path(config.models.embed_model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Embedding model not found: {model_path}\n"
                                    f"Run `motif setup` to download models.")
        self._embedder = Embedder(model_path)
        self._embedder._load()
    - Return self._embedder

get_reranker(self, config: RAGConfig) -> "Reranker"
    - Lazy import: from rag.models.reranker import Reranker
    - If self._reranker is None:
        model_path = Path(config.models.reranker).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Reranker model not found: {model_path}\n"
                                    f"Run `motif setup` to download models.")
        self._reranker = Reranker(model_path)
        self._reranker._load()
    - Return self._reranker

get_llm(self, config: RAGConfig) -> "LLMClient"
    - Lazy import: from rag.generation.llm_client import LLMClient
    - If self._llm is None:
        model_path = Path(config.models.llm_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"LLM not found: {model_path}\n"
                                    f"Run `motif setup` to download models.")
        self._llm = LLMClient(model_path, config)
        self._llm._load()
    - Return self._llm

unload_embedder(self) -> None
    - If self._embedder: self._embedder.unload()
    - self._embedder = None

unload_reranker(self) -> None
    - If self._reranker: self._reranker.unload()
    - self._reranker = None

unload_llm(self) -> None
    - If self._llm: self._llm.unload()
    - self._llm = None

unload_all(self) -> None
    - Call unload_embedder(), unload_reranker(), unload_llm()

after_ingestion(self, config: RAGConfig) -> None
    - On T1: unload_embedder() to free ~550 MB RAM before LLM loads
    - On T2/T3: keep embedder loaded (enough VRAM/RAM)
    - tier = config.resolved_tier
    - if tier == "T1": self.unload_embedder()

status(self) -> dict
    - Return {
        "embedder_loaded": self._embedder is not None and self._embedder.is_loaded(),
        "reranker_loaded": self._reranker is not None and self._reranker.is_loaded(),
        "llm_loaded": self._llm is not None,
      }
```

**Module-level singleton:**
```python
# At the bottom of model_manager.py
_manager = ModelManager()

def get_model_manager() -> ModelManager:
    return _manager
```

All other modules call `get_model_manager()` — never `ModelManager()` directly.

---

## Test Specifications

### `tests/unit/test_chunk_store.py`

```
Fixtures used: minimal_config, tmp_db_root (from conftest.py)

test_insert_and_fetch:
    chunk = Chunk(id="abc123", text="Hello world", source="/docs/test.pdf",
                  filename="test.pdf", source_type="pdf", page=1)
    store.insert(chunk)
    result = store.fetch("abc123")
    assert result.text == "Hello world"
    assert result.page == 1
    assert result.source_type == "pdf"

test_fetch_missing_returns_none:
    result = store.fetch("nonexistent_id")
    assert result is None

test_insert_or_replace:
    chunk = Chunk(id="abc", text="Original", ...)
    store.insert(chunk)
    chunk2 = Chunk(id="abc", text="Updated", ...)
    store.insert(chunk2)
    result = store.fetch("abc")
    assert result.text == "Updated"

test_insert_batch:
    chunks = [Chunk(id=f"c{i}", text=f"text{i}", ...) for i in range(10)]
    store.insert_batch(chunks)
    assert store.count() == 10

test_delete_by_source:
    store.insert(Chunk(id="a", source="/docs/a.pdf", ...))
    store.insert(Chunk(id="b", source="/docs/a.pdf", ...))
    store.insert(Chunk(id="c", source="/docs/b.pdf", ...))
    n = store.delete_by_source("/docs/a.pdf")
    assert n == 2
    assert store.count() == 1

test_count_documents:
    store.insert(Chunk(id="a", source="/docs/a.pdf", ...))
    store.insert(Chunk(id="b", source="/docs/a.pdf", ...))
    store.insert(Chunk(id="c", source="/docs/b.pdf", ...))
    assert store.count_documents() == 2

test_bool_fields_roundtrip:
    chunk = Chunk(id="x", ..., has_table=True, is_ocr=True)
    store.insert(chunk)
    result = store.fetch("x")
    assert result.has_table is True
    assert result.is_ocr is True

test_fetch_batch:
    chunks = [Chunk(id=f"c{i}", ...) for i in range(5)]
    store.insert_batch(chunks)
    results = store.fetch_batch(["c0", "c2", "c4", "nonexistent"])
    assert len(results) == 3  # nonexistent filtered out
```

---

### `tests/unit/test_tracker.py`

```
test_is_indexed_false_for_new_file:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"content")
    assert tracker.is_indexed(path) is False

test_update_and_is_indexed:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"content")
    tracker.update(path, content_hash="abc123", chunk_count=5)
    assert tracker.is_indexed(path) is True

test_get_hash:
    tracker.update(path, content_hash="abc123", chunk_count=5)
    assert tracker.get_hash(path) == "abc123"

test_remove:
    tracker.update(path, content_hash="abc123", chunk_count=5)
    tracker.remove(path)
    assert tracker.is_indexed(path) is False

test_compute_file_hash_deterministic:
    path.write_bytes(b"hello motif")
    h1 = compute_file_hash(path)
    h2 = compute_file_hash(path)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex

test_compute_file_hash_changes_with_content:
    path.write_bytes(b"version 1")
    h1 = compute_file_hash(path)
    path.write_bytes(b"version 2")
    h2 = compute_file_hash(path)
    assert h1 != h2

test_list_all:
    tracker.update(pathA, "hash_a", 3)
    tracker.update(pathB, "hash_b", 7)
    entries = tracker.list_all()
    assert len(entries) == 2
    assert any(e["chunk_count"] == 3 for e in entries)
```

---

### `tests/unit/test_bm25.py`

```
test_add_and_search_basic:
    chunks = [
        Chunk(id="1", text="the cat sat on the mat", ...),
        Chunk(id="2", text="the dog barked loudly", ...),
        Chunk(id="3", text="neural networks for natural language processing", ...),
    ]
    for c in chunks: index.add(c)
    results = index.search("cat mat", top_k=3)
    assert results[0][0] == "1"  # most relevant first

test_search_returns_correct_top_k:
    for i in range(20):
        index.add(Chunk(id=str(i), text=f"document number {i} with unique token tok{i}", ...))
    results = index.search("unique token", top_k=5)
    assert len(results) <= 5

test_search_empty_index:
    results = index.search("anything", top_k=10)
    assert results == []

test_delete:
    index.add(Chunk(id="keep", text="keep this document", ...))
    index.add(Chunk(id="remove", text="remove this document", ...))
    index.delete("remove")
    results = index.search("remove document", top_k=5)
    ids = [r[0] for r in results]
    assert "remove" not in ids

test_count:
    for i in range(7):
        index.add(Chunk(id=str(i), text=f"text {i}", ...))
    assert index.count() == 7

test_persist_and_reload:
    index.add(Chunk(id="persist_test", text="persistent document about oceans", ...))
    index.save()
    index2 = BM25Index(config)  # creates new instance, loads from disk
    results = index2.search("oceans", top_k=3)
    assert results[0][0] == "persist_test"

test_add_duplicate_replaces:
    index.add(Chunk(id="dup", text="original text", ...))
    index.add(Chunk(id="dup", text="replacement text different words", ...))
    assert index.count() == 1
    results = index.search("replacement", top_k=1)
    assert results[0][0] == "dup"

test_zero_score_filtered:
    index.add(Chunk(id="1", text="cats and dogs", ...))
    results = index.search("quantum mechanics", top_k=10)
    # "quantum mechanics" has no overlap — should return empty or score=0 entries
    for _, score in results:
        assert score > 0.0
```

---

## Validation Checklist

Run these commands in order. Phase 1 is complete only when all pass.

```bash
# 1. Verify all package inits are importable
python -c "from rag.storage.chunk_store import ChunkStore; print('ChunkStore OK')"
python -c "from rag.storage.ingestion_tracker import IngestionTracker, compute_file_hash; print('Tracker OK')"
python -c "from rag.retrieval.bm25_index import BM25Index; print('BM25 OK')"
python -c "from rag.models.model_manager import get_model_manager; print('ModelManager OK')"

# 2. Run unit tests
pytest tests/unit/test_chunk_store.py -v
pytest tests/unit/test_tracker.py -v
pytest tests/unit/test_bm25.py -v

# 3. All must pass
pytest tests/unit/test_chunk_store.py tests/unit/test_tracker.py tests/unit/test_bm25.py -v --tb=short
# Expected: XX passed, 0 failed, 0 errors

# 4. Verify no model imports are triggered at import time
python -c "
import rag.storage.chunk_store
import rag.storage.ingestion_tracker
import rag.retrieval.bm25_index
import rag.models.model_manager
print('All Phase 1 modules import without triggering model loads — OK')
"
```

---

## Post-Phase Documentation Updates

When all validation commands pass, update the following files:

**`project-context/progress.md`:**
- Mark Phase 1 plan tasks ✅ (ChunkStore, IngestionTracker, BM25Index, ModelManager)
- Update Phase Status Overview table: Phase 1 → 🔄 In progress (partial)
- Add Metrics Snapshot row: date, Phase 1 partial, storage tests passing

**`project-context/tests.md`:**
- Mark test cases ING-01 through ING-05 (tracker tests) as passing
- Add BM25 test IDs if not present

**Note:** Do NOT mark progress.md Phase 1 as ✅ complete — it requires Phase 2
and Phase 3 to finish.
