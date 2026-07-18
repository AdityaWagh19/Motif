# Phase 4 — Quality & Retrieval Hardening

> **Status:** Not started  
> **Prerequisite:** Phase 3 complete (end-to-end query works, ≥ 70% accuracy on 20 questions)  
> **Model downloads required:** None (uses models already downloaded in Phase 3)  
> **Estimated scope:** 6 files modified, ~800 lines of new implementation

---

## Objective

Push the system from a working baseline to a high-quality RAG system that meets
the 85% faithfulness target. This phase focuses on retrieval quality improvements
(HyDE, semantic chunking, better BM25+dense fusion), document management (/sync,
/remove fully working), and DOCX support.

The 85% faithfulness target is validated using a synthetic evaluation set
generated from the indexed corpus.

**Improvements in this phase:**

| Feature | Benefit | Tiers |
|---|---|---|
| HyDE query expansion | +5–8% recall on complex queries | T2, T3 |
| SemanticChunker | Better chunk boundaries → more coherent passages | T2, T3 |
| DOCX parser | DOCX files can be ingested | All |
| /sync command | Directory sync: add new, remove deleted, re-index changed | All |
| Adjacent chunk merging | Reduces fragmentation in retrieved context | All |
| bge-reranker-base | Higher reranking quality | T3 |

---

## Scope

**In scope:**
- `rag/retrieval/expander.py` — implement HyDE fully
- `rag/ingestion/chunker.py` — add `SemanticChunker`
- `rag/ingestion/parsers/docx.py` — DOCX parser
- `rag/ingestion/parsers/base.py` — update `get_parser()` to include DOCX
- `rag/ingestion/__init__.py` — implement `sync_directory()`
- `rag/commands/sync.py` — remove stub, use real implementation
- `rag/generation/context_builder.py` — add adjacent chunk merging
- `rag/evaluation/test_generator.py` — synthetic QA pair generation
- `tests/unit/test_hyde.py`
- `tests/unit/test_semantic_chunker.py`
- `tests/unit/test_docx_parser.py`
- `tests/integration/test_sync.py`
- `tests/integration/test_eval.py`

**Out of scope:**
- Image parser (Phase 5)
- Audio parser (Phase 5)
- RAGAS full evaluation framework (Phase 6)
- Query cache (Phase 6)
- tantivy BM25 backend (Phase 6)

---

## File Specifications

### `rag/retrieval/expander.py` (full HyDE implementation)

**HyDE (Hypothetical Document Embeddings):**

1. Use the LLM to generate a short hypothetical answer to the query (2–3 sentences)
2. Embed the hypothetical answer (not the query) to search the vector store
3. The hypothesis is closer in embedding space to relevant documents than the query itself

**Routing heuristic — when to use HyDE:**
- Query has > 8 words (short queries are factual — HyDE adds noise)
- Config `query_expansion = "hyde"`
- Tier is T2 or T3 (T1 latency budget too tight — HyDE adds ~1–2s)
- Query does NOT start with who/what/when/where/how many (those are factual lookups)

```python
FACTUAL_PREFIXES = (
    "who is", "who was", "what is", "what are", "what was",
    "when did", "when was", "where is", "where was", "how many",
    "how much", "list ", "name ", "define ",
)

def should_use_hyde(query: str, config: RAGConfig) -> bool:
    if config.retrieval.query_expansion != "hyde":
        return False
    if config.resolved_tier == "T1":
        return False
    words = query.lower().split()
    if len(words) <= 8:
        return False
    query_lower = query.lower()
    if any(query_lower.startswith(p) for p in FACTUAL_PREFIXES):
        return False
    return True


class QueryExpander:

    def expand(
        self,
        query: str,
        config: RAGConfig,
        embedder: "Embedder",
    ) -> Tuple[np.ndarray, str]:
        """
        Expand a query. Returns (query_vector, effective_query_text).

        If HyDE is active:
            - Generate a hypothetical answer using the LLM
            - Embed the hypothetical answer
            - Return (hypothesis_vector, hypothesis_text)
        Else:
            - Embed the original query directly
            - Return (query_vector, query_text)
        """
        if should_use_hyde(query, config):
            llm = get_model_manager().get_llm(config)
            hypothesis = llm.generate(
                HYDE_PROMPT.format(query=query),
                max_tokens=150,
                temperature=0.3,  # slightly higher for creative hypothesis
            )
            hypothesis = hypothesis.strip()
            if not hypothesis:
                # Fallback if LLM returns empty
                vector = embedder.encode(query, prefix="search_query: ")
                return vector, query
            vector = embedder.encode(hypothesis, prefix="search_document: ")
            return vector, hypothesis
        else:
            vector = embedder.encode(query, prefix="search_query: ")
            return vector, query
```

---

### `rag/ingestion/chunker.py` — add `SemanticChunker`

**SemanticChunker** uses the embedder to detect topic boundary shifts.

**Algorithm:**
1. Split text into sentences (same regex as SentenceChunker)
2. Encode each sentence individually
3. Compute cosine similarity between consecutive sentence embeddings
4. Where similarity drops below `threshold` (default 0.3): semantic boundary → new chunk
5. Accumulate sentences within a chunk, enforce max size fallback

```python
class SemanticChunker:
    """
    Splits on semantic boundary shifts detected via embedding cosine distance.
    Falls back to SentenceChunker behaviour when a single semantic chunk exceeds target_tokens.

    Requires: embedder loaded via ModelManager. Only used on T2/T3.
    """

    def __init__(
        self,
        config: "RAGConfig",
        threshold: float = 0.3,
    ) -> None:
        self._config = config
        self._threshold = config.chunking.semantic_threshold
        self._sentence_chunker = SentenceChunker(ChunkerConfig(
            target_tokens=config.chunking.target_tokens,
            overlap_tokens=config.chunking.overlap_tokens,
        ))

    def chunk(self, page: "ParsedPage", source: str, filename: str, source_type: str) -> List[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', page.text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return self._sentence_chunker.chunk(page, source, filename, source_type)

        # Encode all sentences
        embedder = get_model_manager().get_embedder(self._config)
        vecs = embedder.encode_batch(sentences, prefix="search_document: ")

        # Find semantic boundaries
        boundaries = [0]  # always start a chunk at sentence 0
        for i in range(1, len(sentences)):
            sim = float(vecs[i - 1] @ vecs[i])  # cosine sim (vectors are L2-normalised)
            if sim < self._threshold:
                boundaries.append(i)
        boundaries.append(len(sentences))

        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment_text = " ".join(sentences[start:end])
            segment_page = ParsedPage(
                text=segment_text,
                page=page.page,
                section=page.section,
                has_table=page.has_table,
                has_image=page.has_image,
                is_ocr=page.is_ocr,
            )
            # Apply SentenceChunker within each semantic segment
            # (handles the case where one semantic segment is very long)
            sub_chunks = self._sentence_chunker.chunk(segment_page, source, filename, source_type)
            chunks.extend(sub_chunks)

        return chunks

    def chunk_pages(self, pages, source, filename, source_type) -> List[Chunk]:
        all_chunks = []
        for page in pages:
            all_chunks.extend(self.chunk(page, source, filename, source_type))
        return all_chunks
```

**Update `rag/ingestion/__init__.py`** to select chunker based on config:

```python
def _make_chunker(config: RAGConfig):
    if config.chunking.use_semantic and config.resolved_tier in ("T2", "T3"):
        from rag.ingestion.chunker import SemanticChunker
        return SemanticChunker(config)
    else:
        from rag.ingestion.chunker import SentenceChunker, ChunkerConfig
        return SentenceChunker(ChunkerConfig(
            target_tokens=config.chunking.target_tokens,
            overlap_tokens=config.chunking.overlap_tokens,
        ))
```

---

### `rag/ingestion/parsers/docx.py`

**Backend:** `python-docx` (import as `docx`)

**Strategy:**
- Extract paragraphs and tables
- Track heading styles (Heading 1, Heading 2, Heading 3) as section markers
- Convert tables to markdown pipe table format
- Return one ParsedPage per logical section (all content under a heading)

```python
SUPPORTED_EXTENSIONS = [".docx"]

def parse(self, path: Path) -> List[ParsedPage]:
    doc = docx.Document(str(path))
    sections: List[ParsedPage] = []
    current_heading: Optional[str] = None
    current_parts: List[str] = []
    current_has_table = False

    def flush():
        nonlocal current_parts, current_has_table
        if current_parts:
            text = "\n".join(current_parts).strip()
            if text:
                sections.append(ParsedPage(
                    text=text,
                    section=current_heading,
                    has_table=current_has_table,
                ))
        current_parts = []
        current_has_table = False

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            para = docx.text.paragraph.Paragraph(element, doc)
            style_name = para.style.name if para.style else ""
            text = para.text.strip()

            if style_name.startswith("Heading"):
                flush()
                current_heading = text or current_heading
            elif text:
                current_parts.append(text)

        elif tag == "tbl":
            table = docx.table.Table(element, doc)
            md_table = _table_to_markdown(table)
            if md_table:
                current_parts.append(md_table)
                current_has_table = True

    flush()
    return sections


def _table_to_markdown(table) -> str:
    """Convert a python-docx Table to a markdown pipe table string."""
    rows = []
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows) if rows else ""
```

**Update `rag/ingestion/parsers/base.py` `get_parser()`:**

```python
from rag.ingestion.parsers.docx import DOCXParser

for parser_class in [PDFParser, DOCXParser, MarkdownParser]:
    if parser_class.can_parse(path):
        return parser_class()

raise ValueError(
    f"No parser available for '{path.suffix}'. "
    f"Supported: .pdf, .docx, .md, .txt"
)
```

---

### `rag/ingestion/__init__.py` — implement `sync_directory()`

```python
def sync_directory(
    directory: Path,
    config: RAGConfig,
    recursive: bool = False,
    console=None,
) -> SyncResult:
    """
    Synchronise a directory with the knowledge base.

    Algorithm:
    1. Collect all supported files in directory (respecting recursive flag)
    2. Load all tracked files from IngestionTracker
    3. Compute three sets:
       A. New files: on disk but not in tracker
       B. Deleted files: in tracker but not on disk
       C. Changed files: in tracker, on disk, but hash changed
    4. For each deleted file: call remove_document()
    5. For each new or changed file: call ingest_path() for that single file
    6. Return SyncResult with counts
    """
    tracker = IngestionTracker(config)
    supported_exts = {".pdf", ".docx", ".md", ".txt", ".markdown"}

    # Files currently on disk
    if recursive:
        disk_files = {f.resolve() for f in directory.rglob("*")
                      if f.is_file() and f.suffix.lower() in supported_exts}
    else:
        disk_files = {f.resolve() for f in directory.iterdir()
                      if f.is_file() and f.suffix.lower() in supported_exts}

    # Files currently in tracker (within this directory)
    tracked_entries = {
        Path(e["filepath"]): e["content_hash"]
        for e in tracker.list_all()
        if Path(e["filepath"]).parent == directory.resolve()
           or (recursive and _is_subpath(Path(e["filepath"]), directory.resolve()))
    }

    new_files = disk_files - set(tracked_entries.keys())
    deleted_paths = set(tracked_entries.keys()) - disk_files
    changed_files = {
        f for f in disk_files & set(tracked_entries.keys())
        if compute_file_hash(f) != tracked_entries[f]
    }

    added = 0
    removed = 0
    reindexed = 0
    errors = []

    for path in deleted_paths:
        try:
            remove_document(path, config)
            removed += 1
        except Exception as e:
            errors.append(f"remove {path.name}: {e}")

    for path in new_files:
        try:
            result = ingest_path(path, config, recursive=False, console=None)
            added += result.chunks_added
        except Exception as e:
            errors.append(f"ingest {path.name}: {e}")

    for path in changed_files:
        try:
            remove_document(path, config)
            result = ingest_path(path, config, recursive=False, console=None)
            reindexed += result.chunks_added
        except Exception as e:
            errors.append(f"reindex {path.name}: {e}")

    if console:
        console.print(
            f"Sync: [green]+{added}[/green] added  "
            f"[red]-{removed}[/red] removed  "
            f"[yellow]~{reindexed}[/yellow] re-indexed"
        )

    return SyncResult(added=added, removed=removed, reindexed=reindexed, errors=errors)

def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
```

---

### `rag/generation/context_builder.py` — add adjacent chunk merging

Adjacent chunks from the same source and consecutive pages are often split
artificially. Merging them reduces context fragmentation.

```python
def _merge_adjacent_chunks(passages: List[ScoredPassage]) -> List[ScoredPassage]:
    """
    Merge consecutive passages from the same source where page numbers are
    adjacent (N and N+1). Merged passage gets the higher score and method "merged".

    Only merges when both passages are from the same file and same source_type.
    Merged text = passage_A.text + "\n\n" + passage_B.text
    """
    if len(passages) <= 1:
        return passages

    merged = []
    i = 0
    while i < len(passages):
        current = passages[i]
        if i + 1 < len(passages):
            nxt = passages[i + 1]
            can_merge = (
                current.chunk.source == nxt.chunk.source
                and current.chunk.page is not None
                and nxt.chunk.page is not None
                and nxt.chunk.page == current.chunk.page + 1
            )
            if can_merge:
                merged_text = current.chunk.text + "\n\n" + nxt.chunk.text
                merged_chunk = Chunk(
                    id=current.chunk.id,  # keep first chunk's ID for citation
                    text=merged_text,
                    source=current.chunk.source,
                    filename=current.chunk.filename,
                    source_type=current.chunk.source_type,
                    page=current.chunk.page,
                    section=current.chunk.section or nxt.chunk.section,
                    token_count=current.chunk.token_count + nxt.chunk.token_count,
                    indexed_at=current.chunk.indexed_at,
                )
                merged.append(ScoredPassage(
                    chunk=merged_chunk,
                    score=max(current.score, nxt.score),
                    retrieval_method="merged",
                ))
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged
```

Apply `_merge_adjacent_chunks` in `ContextBuilder.build()` before anti-middle ordering:

```python
def build(self, passages, query, history, config):
    selected = self._apply_token_budget(passages, query, history, config)
    selected = _merge_adjacent_chunks(selected)  # NEW: merge adjacent
    ordered = _anti_middle_order(selected)
    prompt = build_prompt(query, ordered, history)
    return prompt, ordered
```

---

### `rag/evaluation/test_generator.py`

**Purpose:** Generate synthetic question-answer pairs from indexed chunks for
offline evaluation. Questions are generated by the LLM, answers are the chunk text.

```python
def create_eval_dataset(
    config: RAGConfig,
    n: int = 50,
    output_path: Optional[Path] = None,
) -> List[dict]:
    """
    Generate a synthetic evaluation dataset.

    Algorithm:
    1. Sample n chunks from ChunkStore (random, stratified by source_type)
    2. For each chunk, prompt the LLM:
         "Generate one specific, answerable question about this text:
          <chunk text>
          Question:"
    3. Record {question, ground_truth_answer (chunk text), source, source_type}
    4. Save to output_path as JSON if provided
    5. Return list of dicts

    The dataset is used by ragas_runner.py (Phase 6) for offline evaluation.
    """
    store = ChunkStore(config)
    llm = get_model_manager().get_llm(config)

    # Sample chunks
    conn = sqlite3.connect(str(config.db_root / "chunks.db"))
    rows = conn.execute(
        "SELECT id, text, filename, source_type FROM chunks ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    conn.close()

    QUESTION_PROMPT = (
        "Generate one specific, answerable question whose answer is contained in this text. "
        "The question should be clear and factual.\n\nText:\n{text}\n\nQuestion:"
    )

    dataset = []
    for row in rows:
        chunk_id, text, filename, source_type = row
        question = llm.generate(
            QUESTION_PROMPT.format(text=text[:800]),  # truncate for prompt safety
            max_tokens=60,
            temperature=0.4,
        ).strip()
        if question:
            dataset.append({
                "question": question,
                "ground_truth": text,
                "source": filename,
                "source_type": source_type,
                "chunk_id": chunk_id,
            })

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    return dataset
```

---

## Test Specifications

### `tests/unit/test_hyde.py`

```
test_should_use_hyde_false_for_t1:
    config.resolved_tier = "T1"
    config.retrieval.query_expansion = "hyde"
    assert should_use_hyde("explain the methodology used in this paper", config) is False

test_should_use_hyde_false_for_short_query:
    config.resolved_tier = "T2"
    config.retrieval.query_expansion = "hyde"
    assert should_use_hyde("what is X", config) is False

test_should_use_hyde_false_for_factual_prefix:
    config.resolved_tier = "T2"
    config.retrieval.query_expansion = "hyde"
    assert should_use_hyde("who is the author of this document", config) is False

test_should_use_hyde_true_for_complex:
    config.resolved_tier = "T2"
    config.retrieval.query_expansion = "hyde"
    assert should_use_hyde("explain the relationship between the retrieval methods and accuracy", config) is True

test_should_use_hyde_false_when_disabled:
    config.resolved_tier = "T2"
    config.retrieval.query_expansion = "none"
    assert should_use_hyde("explain the relationship between retrieval and accuracy", config) is False
```

### `tests/unit/test_semantic_chunker.py`

```
test_semantic_chunker_splits_topic_shifts:
    # Provide text with two clearly different topics
    text = (
        "The Eiffel Tower is located in Paris, France. It was built in 1889. "
        "It stands 330 metres tall. " * 5 +
        "Python is a programming language. It supports multiple paradigms. "
        "Python was created by Guido van Rossum. " * 5
    )
    # Mock embedder to return high similarity within topic, low across topics
    [verify chunker produces at least 2 chunks for this input]

test_semantic_chunker_falls_back_for_short_text:
    short = ParsedPage(text="One sentence only.")
    chunks = sem_chunker.chunk(short, "/f.md", "f.md", "md")
    assert len(chunks) == 1
```

### `tests/unit/test_docx_parser.py`

```
test_docx_parser_extracts_headings_as_sections:
    # Create a minimal DOCX programmatically with python-docx
    # Add Heading 1 "Introduction", paragraph text, Heading 2 "Methods"
    pages = DOCXParser().parse(docx_path)
    sections = [p.section for p in pages if p.section]
    assert "Introduction" in sections

test_docx_parser_converts_table_to_markdown:
    # DOCX with a 2×3 table
    pages = DOCXParser().parse(docx_with_table)
    assert any("|" in p.text for p in pages)  # pipe tables present
    assert any(p.has_table for p in pages)

test_docx_parser_extension:
    assert DOCXParser.can_parse(Path("doc.docx")) is True
    assert DOCXParser.can_parse(Path("doc.pdf")) is False
```

### `tests/integration/test_sync.py`

```
@pytest.mark.slow
def test_sync_detects_new_file(minimal_config, tmp_path):
    # Ingest an initial directory
    d = tmp_path / "corpus"
    d.mkdir()
    f1 = d / "doc1.md"; f1.write_text("First document text " * 20)
    ingest_path(f1, config=minimal_config, recursive=False, console=None)

    # Add a new file
    f2 = d / "doc2.md"; f2.write_text("Second document text " * 20)
    result = sync_directory(d, config=minimal_config, recursive=False, console=None)
    assert result.added > 0
    assert ChunkStore(minimal_config).count_documents() == 2

@pytest.mark.slow
def test_sync_detects_deleted_file(minimal_config, tmp_path):
    d = tmp_path / "corpus"; d.mkdir()
    f1 = d / "doc1.md"; f1.write_text("Document to be deleted " * 20)
    f2 = d / "doc2.md"; f2.write_text("Document that stays " * 20)
    ingest_path(d, config=minimal_config, recursive=False, console=None)
    assert ChunkStore(minimal_config).count_documents() == 2
    f1.unlink()
    result = sync_directory(d, config=minimal_config, recursive=False, console=None)
    assert result.removed > 0
    assert ChunkStore(minimal_config).count_documents() == 1

@pytest.mark.slow
def test_sync_detects_changed_file(minimal_config, tmp_path):
    d = tmp_path / "corpus"; d.mkdir()
    f = d / "doc.md"; f.write_text("Original content here " * 20)
    ingest_path(f, config=minimal_config, recursive=False, console=None)
    old_count = ChunkStore(minimal_config).count()
    f.write_text("Completely new different content about databases " * 30)
    result = sync_directory(d, config=minimal_config, recursive=False, console=None)
    assert result.reindexed > 0
```

---

## Validation Checklist

```bash
# 1. Imports
python -c "from rag.retrieval.expander import QueryExpander, should_use_hyde; print('HyDE OK')"
python -c "from rag.ingestion.parsers.docx import DOCXParser; print('DOCX OK')"

# 2. Unit tests
pytest tests/unit/test_hyde.py tests/unit/test_semantic_chunker.py tests/unit/test_docx_parser.py -v

# 3. Integration tests
pytest tests/integration/test_sync.py -v -m slow

# 4. Functional REPL verification
# motif
# /ingest ./project-context -r   [should use semantic chunker on T2/T3]
# /sync ./project-context        [should report 0 new, 0 removed, 0 changed]
# [modify a file, then] /sync ./project-context  [should report 1 re-indexed]

# 5. Accuracy evaluation
# python -c "
# from rag.config import load_config
# from rag.evaluation.test_generator import create_eval_dataset
# config = load_config()
# dataset = create_eval_dataset(config, n=20, output_path=None)
# print(f'Generated {len(dataset)} questions')
# print(dataset[0])
# "

# 6. Manual accuracy check — run the 20 generated questions through the pipeline
# Target: >= 85% faithfulness on T2/T3  (>= 75% on T1)
# Record in progress.md Metrics Snapshots
```

---

## Post-Phase Documentation Updates

**`project-context/progress.md`:**
- Mark all Phase 2 (Quality) tasks ✅
- Update Phase Status Overview: Phase 2 → ✅ Done
- Add Metrics Snapshot row: faithfulness ≥ 85% (T2/T3), date

**`project-context/tests.md`:**
- Mark RET-05, RET-06 (HyDE, semantic chunking) ✅
- Mark ING-15, ING-16 (DOCX parser) ✅
- Mark ING-17, ING-18 (/sync add/remove/reindex) ✅

**Deferred Decisions Log:**
- "HyDE vs multi-query" — resolve: HyDE adopted based on Phase 4 accuracy gains
- "bge-reranker-base for T2" — resolve after measuring accuracy delta
