# Phase 7 — Optional Enhancements

> **Status:** Not planned until Phase 6 complete  
> **Prerequisite:** Phase 6 complete (all NFRs met, RAGAS baseline saved)  
> **Model downloads required:** Varies per feature (noted per item)  
> **Estimated scope:** Independent features — implement in any order

---

## Objective

Optional features that improve specific use cases but are not required for the
core RAG system. These are implemented based on user feedback after Phase 6.
Each item is self-contained and does not gate any other item.

---

## Features

### 7-A: RAPTOR Hierarchical Indexing

**Use case:** Document corpora with > 500 pages where high-level conceptual
questions perform poorly on chunk-level retrieval.

**What it does:**
1. After initial ingestion, cluster all chunks using k-means (k = √n)
2. For each cluster, prompt the LLM to write a 1-paragraph "summary chunk"
3. Index summary chunks alongside regular chunks (different Qdrant payload field: `is_summary=True`)
4. At query time: retrieve from both levels, merge, deduplicate

**Files to create:**
- `rag/ingestion/raptor.py` — clustering + summary generation
- Update `rag/pipeline.py` — add `use_raptor` flag to `answer()`

**When to enable:**
```toml
[retrieval]
use_raptor = true   # only if > 500 pages indexed
```

**Validation:**
- Run RAGAS on conceptual questions: e.g., "What is the overall thesis of this corpus?"
- RAPTOR should improve answer_relevancy by ≥ 3% on high-level questions

---

### 7-B: Parent-Document Retrieval

**Use case:** Chunks are short (e.g., 128 tokens) for precise retrieval, but the
LLM needs more context to give a complete answer.

**What it does:**
- At ingestion: store both "child" chunks (128 tokens) and "parent" chunks (512 tokens)
  in ChunkStore (parent_id field links them)
- At retrieval: retrieve by child chunk (precision), expand to parent chunk (context)
- Pass parent chunk text to LLM, cite child chunk location

**Files to create/modify:**
- Update `rag/types.py` — add `parent_id: Optional[str] = None` to Chunk
- Update `rag/ingestion/chunker.py` — add `ParentChunker`
- Update `rag/storage/chunk_store.py` — add `fetch_parent()` method
- Update `rag/pipeline.py` — add `use_parent_docs` flag

**Cost:** Storage doubles (child + parent chunks). Only recommended if
retrieval recall is measured to be insufficient.

---

### 7-C: FLARE Iterative Retrieval

**Use case:** Complex multi-hop questions (e.g., "Compare the methods in
document A with the results in document B") that require multiple retrieval steps.

**What it does:**
- During generation, when the LLM is about to generate an uncertain token
  (detected via low logit probability), pause and retrieve fresh context
- Re-inject retrieved passages and continue generation

**Blocker:** Requires token-level logit access from llama-cpp-python.
As of Phase 7, verify if `Llama.__call__` exposes `logprobs` in the output.
If available: implement. If not: defer.

**Files to create:**
- `rag/generation/flare.py` — iterative retrieval controller

**Validation:** Run RAGAS on a multi-hop eval set (requires creating one).
FLARE should improve faithfulness on multi-hop by ≥ 5%.

---

### 7-D: REST API Wrapper

**Use case:** Use Motif as a backend service for other tools, scripts, or
a desktop GUI.

**Backend:** FastAPI

**Endpoints:**

```
POST /query
    Body: {"query": str, "session_id": str | null}
    Returns: {"answer": str, "citations": [...], "latency_ms": float}

POST /ingest
    Body: {"path": str, "recursive": bool}
    Returns: {"files_processed": int, "chunks_added": int, "errors": [...]}

GET /status
    Returns: {"documents": int, "chunks": int, "tier": str}

DELETE /document
    Body: {"path": str}
    Returns: {"chunks_removed": int}

POST /sync
    Body: {"directory": str, "recursive": bool}
    Returns: SyncResult
```

**Server lifecycle:**
- Server holds a single `QueryPipeline` instance across requests (models stay warm)
- Session management: map `session_id` → `Session` object
- Run with: `motif serve --host 127.0.0.1 --port 8765`

**Files to create:**
- `rag/api/server.py` — FastAPI app
- `rag/api/models.py` — Pydantic request/response models
- `rag/commands/serve.py` — `/serve` slash command and `motif serve` entry point

**Validation:**
```bash
motif serve &
curl -s -X POST http://localhost:8765/status | python -m json.tool
# Expected: {"documents": N, "chunks": M, "tier": "T1"}
```

---

### 7-E: Desktop GUI (Tauri)

**Use case:** Users who prefer a graphical interface over a terminal.

**Backend:** Rust + Tauri  
**Frontend:** HTML/CSS/JS (no framework — keeps bundle size minimal)

**Features:**
- Chat interface (messages + citations with clickable links)
- File/folder drag-and-drop for ingestion
- Progress bars for ingestion
- Settings panel (config.toml editor)

**Implementation path:**
1. REST API (7-D) must be done first
2. Tauri app calls the REST API
3. Tauri shell command for `motif serve` lifecycle management

**Files to create:**
- `gui/` — Tauri project directory
- `gui/src-tauri/` — Rust Tauri app
- `gui/src/` — HTML/JS frontend

---

### 7-F: NOUGAT Academic PDF Parser

**Use case:** Academic PDFs with complex layouts, mathematical formulas, tables,
and multi-column text that PyMuPDF cannot parse reliably.

**Backend:** `nougat-ocr` (Facebook Research)

**Install:** `pip install nougat-ocr` — large install, ~3 GB VRAM required.
T3 opt-in only.

```bash
motif setup --tier T3 --nougat   # downloads nougat model
```

**Activation heuristic:**
```python
def is_academic_pdf(path: Path) -> bool:
    """Heuristic: detect if PDF looks like an academic paper."""
    doc = fitz.open(str(path))
    first_page_text = doc[0].get_text("text") if len(doc) > 0 else ""
    doc.close()
    return any(marker in first_page_text.lower()
               for marker in ["abstract", "introduction", "references", "arxiv"])
```

If `is_academic_pdf()` returns True AND `config.parsers.use_nougat` is True:
- Use nougat instead of PyMuPDF

**Files to create:**
- `rag/ingestion/parsers/nougat_parser.py`
- Update `get_parser()` to check for nougat eligibility

---

## Tracking Notes

- Each 7-X item is tracked separately in `progress.md` Phase 5
- No 7-X item blocks any other 7-X item
- Implement in this priority order based on user demand:
  1. REST API (7-D) — most useful for integration
  2. Parent-Document Retrieval (7-B) — measurable quality gain
  3. RAPTOR (7-A) — for large corpus users
  4. FLARE (7-C) — pending llama.cpp logit API
  5. GUI (7-E) — depends on 7-D
  6. NOUGAT (7-F) — narrow use case

---

## Post-Phase Documentation Updates

For each 7-X feature implemented:

**`project-context/progress.md`:**
- Mark that specific Phase 5 subtask ✅
- Add relevant RAGAS metric improvement to Metrics Snapshots

**`README.md`:**
- Update "Features" section to list newly available capabilities
- Update "Setup" if a new `motif setup --flag` option was added
