"""
rag/ingestion/__init__.py — Public API for the ingestion package.

Commands layer imports ONLY from here — never from internal ingestion modules:

    from rag.ingestion import ingest_path, remove_document

Pipeline flow for ingest_path():
    File path
       │
       ▼
    Parser (PDFParser | MarkdownParser)  →  List[ParsedPage]
       │
       ▼
    SentenceChunker                      →  List[Chunk]
       │
       ▼
    Deduplicator                         →  List[Chunk] (near-dups removed)
       │
       ▼
    Embedder.encode_batch()              →  (N, 768) float32 vectors
       │
       ├──► ChunkStore.insert_batch()    →  SQLite  (text + metadata)
       ├──► BM25Index.add_batch()        →  lexical index
       ├──► VectorStore.upsert_batch()   →  Qdrant HNSW
       └──► IngestionTracker.update()    →  file hash + chunk count

Dependency graph position (runtime):
    __init__  →  parsers.base, parsers.pdf, parsers.markdown
    __init__  →  chunker.py
    __init__  →  deduplicator.py
    __init__  →  rag.storage.chunk_store
    __init__  →  rag.storage.ingestion_tracker
    __init__  →  rag.retrieval.bm25_index
    __init__  →  rag.retrieval.vector_store
    __init__  →  rag.models.model_manager
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from rag.types import IngestResult, SyncResult

if TYPE_CHECKING:
    from rich.console import Console
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

# File extensions handled by the Phase 2 parsers.
_SUPPORTED_EXTENSIONS = frozenset([".pdf", ".md", ".txt", ".markdown"])

# Source-type string derived from file extension.
_EXT_TO_SOURCE_TYPE: dict = {
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_files(path: Path, recursive: bool) -> List[Path]:
    """Return all supported files under *path* (file or directory)."""
    if path.is_file():
        return [path] if path.suffix.lower() in _SUPPORTED_EXTENSIONS else []

    if recursive:
        candidates = [f for f in path.rglob("*") if f.is_file()]
    else:
        candidates = [f for f in path.iterdir() if f.is_file()]

    return [f for f in candidates if f.suffix.lower() in _SUPPORTED_EXTENSIONS]


def _chunk_to_payload(chunk) -> dict:  # type: ignore[return]
    """Extract Qdrant payload fields from a Chunk (text is stored in ChunkStore)."""
    return {
        "source": chunk.source,
        "filename": chunk.filename,
        "source_type": chunk.source_type,
        "page": chunk.page,
        "section": chunk.section,
        "has_table": chunk.has_table,
        "has_image": chunk.has_image,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_path(
    path: Path,
    config: "RAGConfig",
    recursive: bool = False,
    console: "Console | None" = None,
) -> IngestResult:
    """
    Ingest all supported documents found at *path* into the knowledge base.

    Supported types (Phase 2): .pdf, .md, .txt, .markdown
    Phase 4 adds: .docx
    Phase 5 adds: images, audio

    Behaviour:
    - If path is a file: ingest that single file.
    - If path is a directory and recursive=False: ingest top-level files only.
    - If path is a directory and recursive=True: ingest all files recursively.
    - Files already indexed with unchanged content hash are skipped.
    - Files whose hash has changed are re-indexed (old chunks deleted first).

    Args:
        path:      Target file or directory (absolute, validated by caller).
        config:    Loaded RAGConfig.
        recursive: If True, recurse into subdirectories.
        console:   Rich Console for progress output. None → silent.

    Returns:
        IngestResult(files_processed, chunks_added, files_skipped, errors)
    """
    from rag.ingestion.parsers.base import get_parser
    from rag.ingestion.chunker import SentenceChunker, ChunkerConfig
    from rag.ingestion.deduplicator import Deduplicator
    from rag.storage.chunk_store import ChunkStore
    from rag.storage.ingestion_tracker import IngestionTracker, compute_file_hash
    from rag.retrieval.bm25_index import BM25Index
    from rag.retrieval.vector_store import VectorStore
    from rag.models.model_manager import get_model_manager

    files = _collect_files(path, recursive)
    if not files and console:
        console.print(
            f"[yellow]No supported files found at:[/yellow] {path}\n"
            f"Supported types: .pdf .md .txt .markdown"
        )
        return IngestResult(
            files_processed=0,
            chunks_added=0,
            files_skipped=0,
            errors=[],
        )

    # Initialise all stores
    tracker = IngestionTracker(config)
    chunk_store = ChunkStore(config)
    bm25 = BM25Index(config)
    vector_store = VectorStore(config)
    model_manager = get_model_manager()
    embedder = model_manager.get_embedder(config)

    # Select chunker: SemanticChunker on T2/T3, SentenceChunker on T1.
    use_semantic = getattr(config.chunking, "use_semantic", False)
    if use_semantic:
        from rag.ingestion.semantic_chunker import SemanticChunker
        chunker = SemanticChunker(config, embedder)  # type: ignore[assignment]
        if console:
            console.print("[dim]Using semantic chunker (T2/T3).[/dim]")
    else:
        chunker = SentenceChunker(
            ChunkerConfig(
                target_tokens=getattr(config.chunking, "target_tokens", 512),
                overlap_tokens=getattr(config.chunking, "overlap_tokens", 64),
            )
        )

    deduplicator = Deduplicator()

    files_processed = 0
    chunks_added = 0
    files_skipped = 0
    errors: List[str] = []

    total = len(files)
    for idx, file in enumerate(files, start=1):
        source = str(file.resolve())
        source_type = _EXT_TO_SOURCE_TYPE.get(file.suffix.lower(), "txt")

        if console:
            console.print(
                f"[dim][{idx}/{total}][/dim] {file.name}",
                end="  ",
            )

        try:
            file_hash = compute_file_hash(file)

            # ── Deduplication check ──────────────────────────────────────────
            if tracker.is_indexed(file):
                if tracker.get_hash(file) == file_hash:
                    # Unchanged — skip
                    if console:
                        console.print("[dim]skipped (unchanged)[/dim]")
                    files_skipped += 1
                    continue
                else:
                    # Modified — remove old version first
                    if console:
                        console.print("[yellow]changed — re-indexing…[/yellow]", end="  ")
                    remove_document(file, config)

            # ── Parse ────────────────────────────────────────────────────────
            parser = get_parser(file, config)
            pages = parser.parse(file)
            if not pages:
                log.warning("No pages extracted from %s — skipping.", file.name)
                if console:
                    console.print("[yellow]no content — skipped[/yellow]")
                continue

            # ── Chunk ────────────────────────────────────────────────────────
            chunks = chunker.chunk_pages(
                pages,
                source=source,
                filename=file.name,
                source_type=source_type,
            )

            # ── Deduplicate (within-document) ────────────────────────────────
            chunks = deduplicator.filter(chunks)
            deduplicator.reset()  # reset between documents

            if not chunks:
                log.warning("All chunks deduplicated for %s — skipping.", file.name)
                if console:
                    console.print("[yellow]all chunks were duplicates — skipped[/yellow]")
                continue

            # ── Embed ────────────────────────────────────────────────────────
            vectors = embedder.encode_batch(
                [c.text for c in chunks],
                prefix="search_document: ",
            )

            # ── Store ─────────────────────────────────────────────────────────
            chunk_store.insert_batch(chunks)
            bm25.add_batch(chunks)
            payloads = [_chunk_to_payload(c) for c in chunks]
            vector_store.upsert_batch(
                [c.id for c in chunks],
                vectors,
                payloads,
            )
            tracker.update(file, file_hash, len(chunks))

            files_processed += 1
            chunks_added += len(chunks)

            if console:
                console.print(
                    f"[green]✓[/green] {len(chunks)} chunk(s)"
                )

        except Exception as exc:
            err_msg = f"{file.name}: {exc}"
            log.exception("Ingestion error for %s", file)
            errors.append(err_msg)
            if console:
                console.print(f"[red]error:[/red] {exc}")

    # Persist BM25 index once after all files (atomic write)
    bm25.save()

    # T1 memory policy: unload embedder before LLM is loaded
    model_manager.after_ingestion(config)

    return IngestResult(
        files_processed=files_processed,
        chunks_added=chunks_added,
        files_skipped=files_skipped,
        errors=errors,
    )


def remove_document(path: Path, config: "RAGConfig") -> int:
    """
    Remove a document and all its indexed chunks from the knowledge base.

    The document file itself is NOT deleted from disk.

    Steps:
    1. Collect chunk IDs from ChunkStore (needed for BM25 delete).
    2. Delete from ChunkStore (SQLite).
    3. Delete from BM25Index and persist.
    4. Delete from VectorStore (Qdrant).
    5. Remove file record from IngestionTracker.

    Args:
        path:   Path to the source document (used to derive the source string).
        config: Loaded RAGConfig.

    Returns:
        Number of chunks removed (from ChunkStore).
    """
    from rag.storage.chunk_store import ChunkStore
    from rag.storage.ingestion_tracker import IngestionTracker
    from rag.retrieval.bm25_index import BM25Index
    from rag.retrieval.vector_store import VectorStore

    source = str(path.resolve())

    chunk_store = ChunkStore(config)
    bm25 = BM25Index(config)
    vector_store = VectorStore(config)
    tracker = IngestionTracker(config)

    # Collect IDs before deletion (BM25 delete requires them)
    chunks = chunk_store.fetch_by_source(source)
    chunk_ids = [c.id for c in chunks]

    # Delete from all stores
    n = chunk_store.delete_by_source(source)
    if chunk_ids:
        bm25.delete_by_source(source, chunk_ids)
        bm25.save()
    vector_store.delete_by_source(source)
    tracker.remove(path)

    log.info("Removed document %s (%d chunks).", path.name, n)
    return n


def sync_directory(
    directory: Path,
    config: "RAGConfig",
    recursive: bool = False,
    console: "Console | None" = None,
) -> SyncResult:
    """
    Synchronise a directory with the knowledge base.

    Phase 4 full implementation:
    - Files present on disk but absent from the index → ingest.
    - Files in the index but deleted from disk → remove.
    - Files whose content hash has changed → remove then re-ingest.

    Args:
        directory: Target directory (must exist, validated by caller).
        config:    Loaded RAGConfig.
        recursive: If True, recurse into subdirectories.
        console:   Rich Console for progress output.

    Returns:
        SyncResult(added, removed, reindexed, errors)
    """
    from rag.storage.ingestion_tracker import IngestionTracker, compute_file_hash

    tracker = IngestionTracker(config)
    indexed = {Path(r["filepath"]): r["content_hash"] for r in tracker.list_all()}

    disk_files = {f.resolve(): f for f in _collect_files(directory, recursive)}
    disk_paths = set(disk_files.keys())
    indexed_paths = set(indexed.keys())

    # Files on disk that are NOT indexed → ingest
    to_add = disk_paths - indexed_paths

    # Files indexed but MISSING from disk → remove
    to_remove = indexed_paths - disk_paths

    # Files that exist on both sides but have a different hash → reindex
    common = disk_paths & indexed_paths
    to_reindex: List[Path] = []
    for p in common:
        try:
            current_hash = compute_file_hash(disk_files[p])
            if current_hash != indexed.get(p):
                to_reindex.append(p)
        except OSError:
            pass  # If we can't read the file, skip it

    added = 0
    removed = 0
    reindexed = 0
    errors: List[str] = []

    # Remove stale entries
    for p in to_remove:
        try:
            removed += 1
            remove_document(p, config)
            if console:
                console.print(f"  [red]removed[/red]  {p.name}")
        except Exception as exc:
            errors.append(f"remove {p.name}: {exc}")
            if console:
                console.print(f"  [red]error removing[/red] {p.name}: {exc}")

    # Ingest new files
    new_files = [disk_files[p] for p in to_add]
    for f in new_files:
        try:
            result = ingest_path(f, config, recursive=False, console=console)
            added += result.files_processed
            if result.errors:
                errors.extend(result.errors)
        except Exception as exc:
            errors.append(f"add {f.name}: {exc}")

    # Reindex changed files
    for p in to_reindex:
        try:
            result = ingest_path(disk_files[p], config, recursive=False, console=console)
            reindexed += result.files_processed
            if result.errors:
                errors.extend(result.errors)
            if console:
                console.print(f"  [yellow]reindexed[/yellow] {p.name}")
        except Exception as exc:
            errors.append(f"reindex {p.name}: {exc}")

    return SyncResult(
        added=added,
        removed=removed,
        reindexed=reindexed,
        errors=errors,
    )
