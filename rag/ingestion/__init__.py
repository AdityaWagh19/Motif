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

# File extensions handled by the ingestion parsers.
# All five parsers (PDF, DOCX, Markdown, Image, Audio) are registered here.
_SUPPORTED_EXTENSIONS = frozenset([
    # Text documents
    ".pdf", ".docx",
    ".md", ".txt", ".markdown",
    # Images (OCR via PaddleOCR; captioning via moondream2 on T3)
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
    # Audio (transcription via Whisper)
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
])

# Source-type string derived from file extension.
_EXT_TO_SOURCE_TYPE: dict = {
    # Documents
    ".pdf":      "pdf",
    ".docx":     "docx",
    ".md":       "md",
    ".markdown": "md",
    ".txt":      "txt",
    # Images
    ".png":      "image",
    ".jpg":      "image",
    ".jpeg":     "image",
    ".webp":     "image",
    ".bmp":      "image",
    ".tiff":     "image",
    # Audio
    ".mp3":      "audio",
    ".wav":      "audio",
    ".m4a":      "audio",
    ".flac":     "audio",
    ".ogg":      "audio",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_files(path: Path, recursive: bool) -> list[Path]:
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
    config: RAGConfig,
    recursive: bool = False,
    console: Console | None = None,
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
    from rag.ingestion.chunker import ChunkerConfig, SentenceChunker
    from rag.ingestion.deduplicator import Deduplicator
    from rag.ingestion.parsers.base import get_parser
    from rag.models.model_manager import get_model_manager
    from rag.retrieval.bm25_index import BM25Index
    from rag.retrieval.vector_store import VectorStore
    from rag.storage.chunk_store import ChunkStore
    from rag.storage.ingestion_tracker import IngestionTracker, compute_file_hash

    files = _collect_files(path, recursive)
    if not files and console:
        console.print(
            f"[yellow]No supported files found at:[/yellow] {path}\n"
            f"Supported types: .pdf .docx .md .txt "  
            f".png .jpg .jpeg .webp .bmp .tiff "  
            f".mp3 .wav .m4a .flac .ogg"
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

    # Select chunker: ParentChunker on 7-B, SemanticChunker on T2/T3, SentenceChunker on T1.
    use_semantic = getattr(config.chunking, "use_semantic", False)
    use_parent_docs = getattr(config.retrieval, "use_parent_docs", False)
    
    if use_parent_docs:
        from rag.ingestion.chunker import ParentChunker, ParentChunkerConfig
        chunker = ParentChunker(ParentChunkerConfig())  # type: ignore[assignment]
        if console:
            console.print("[dim]Using parent-document retrieval mode.[/dim]")
    elif use_semantic:
        from rag.ingestion.semantic_chunker import SemanticChunker
        chunker = SemanticChunker(config, embedder)  # type: ignore[assignment]
        if console:
            console.print("[dim]Using semantic document chunking.[/dim]")
    else:
        chunker = SentenceChunker(
            ChunkerConfig(
                target_tokens=getattr(config.chunking, "target_tokens", 512),
                overlap_tokens=getattr(config.chunking, "overlap_tokens", 64),
            )
        )

    deduplicator = Deduplicator()
    from rag.storage.transaction_manager import StorageTransactionManager
    tx_manager = StorageTransactionManager(config)

    files_processed = 0
    chunks_added = 0
    files_skipped = 0
    errors: list[str] = []

    total = len(files)
    
    import threading
    import queue
    from concurrent.futures import ThreadPoolExecutor

    # Queue for decoupling CPU-bound parsing from GPU-bound embedding
    embed_queue: queue.Queue = queue.Queue()
    
    def embed_and_store_worker():
        nonlocal files_processed, chunks_added
        while True:
            item = embed_queue.get()
            if item is None:
                embed_queue.task_done()
                break
                
            file, file_hash, chunks, indexable_chunks = item
            
            try:
                def _format_for_embedding(c) -> str:
                    ext = getattr(c, "source_type", "").lower()
                    prefix = ""
                    if ext == "audio":
                        prefix = "[MODALITY: AUDIO RECORDING] "
                    elif ext == "image":
                        prefix = "[MODALITY: IMAGE / DIAGRAM] "
                    elif ext == "pdf":
                        prefix = "[MODALITY: PDF DOCUMENT] "
                    elif ext in ("doc", "docx"):
                        prefix = "[MODALITY: WORD DOCUMENT] "
                    return prefix + c.text

                # ── Embed (GPU-bound) ────────────────────────────────────────────
                vectors = embedder.encode_batch(
                    [_format_for_embedding(c) for c in indexable_chunks],
                    prefix="search_document: ",
                )

                # ── Store (I/O-bound) ─────────────────────────────────────────────
                payloads = [_chunk_to_payload(c) for c in indexable_chunks]
                tx_manager.execute_ingest(
                    file_path=file,
                    file_hash=file_hash,
                    chunks=chunks,
                    indexable_chunks=indexable_chunks,
                    vectors=vectors,
                    payloads=payloads,
                    chunk_store=chunk_store,
                    tracker=tracker,
                    bm25=bm25,
                    vector_store=vector_store,
                )

                files_processed += 1
                chunks_added += len(chunks)

                if console:
                    console.print(f"[green]OK[/green] {file.name} ({len(chunks)} chunks)")
            except Exception as exc:
                user_err = f"Could not embed/store {file.name}: {exc}"
                log.exception("Ingestion store error for %s", file)
                errors.append(user_err)
                if console:
                    console.print(f"[red]failed:[/red] {file.name} - {exc}")
            finally:
                embed_queue.task_done()

    # Start the single GPU/Storage worker thread
    worker_thread = threading.Thread(target=embed_and_store_worker, daemon=True)
    worker_thread.start()

    def process_file_cpu(idx: int, file: Path):
        nonlocal files_skipped
        source = str(file.resolve())
        source_type = _EXT_TO_SOURCE_TYPE.get(file.suffix.lower(), "txt")

        try:
            file_hash = compute_file_hash(file)

            # ── Deduplication check ──────────────────────────────────────────
            if tracker.is_indexed(file):
                if tracker.get_hash(file) == file_hash:
                    if console:
                        console.print(f"[dim][{idx}/{total}][/dim] {file.name} [dim]skipped (unchanged)[/dim]")
                    files_skipped += 1
                    return
                else:
                    if console:
                        console.print(f"[dim][{idx}/{total}][/dim] {file.name} [yellow]changed — re-indexing…[/yellow]")
                    remove_document(
                        file, 
                        config, 
                        chunk_store=chunk_store, 
                        bm25=bm25, 
                        vector_store=vector_store, 
                        tracker=tracker
                    )
            else:
                if console:
                    console.print(f"[dim][{idx}/{total}][/dim] {file.name} [dim]parsing...[/dim]")

            # ── Parse (CPU-bound) ────────────────────────────────────────────
            parser = get_parser(file, config)
            pages = parser.parse(file)
            if not pages:
                log.warning("No pages extracted from %s — skipping.", file.name)
                return

            # ── Chunk (CPU-bound) ────────────────────────────────────────────
            chunks = chunker.chunk_pages(
                pages,
                source=source,
                filename=file.name,
                source_type=source_type,
            )

            # ── Deduplicate (within-document) ────────────────────────────────
            # Create a fresh deduplicator for this thread
            from rag.ingestion.deduplicator import Deduplicator
            local_dedup = Deduplicator()
            chunks = local_dedup.filter(chunks)

            if not chunks:
                log.warning("All chunks deduplicated for %s — skipping.", file.name)
                return
                
            if use_parent_docs:
                indexable_chunks = [c for c in chunks if c.parent_id is not None]
            else:
                indexable_chunks = chunks

            # Send to GPU/Storage worker
            embed_queue.put((file, file_hash, chunks, indexable_chunks))

        except Exception as exc:
            user_err = f"Could not parse {file.name}: {exc}"
            log.exception("Ingestion parse error for %s", file)
            errors.append(user_err)
            if console:
                console.print(f"[red]failed:[/red] {file.name} - {exc}")

    try:
        # Use ThreadPoolExecutor for concurrent parsing
        max_workers = getattr(config.chunking, "num_workers", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, file in enumerate(files, start=1):
                executor.submit(process_file_cpu, idx, file)
        
        # Wait for all background embeddings to finish
        embed_queue.put(None)
        worker_thread.join()

        # ── Hierarchical Indexing ──────────────────────────
        if getattr(config.retrieval, "use_raptor", False) and files_processed > 0:
            from rag.ingestion.raptor import build_raptor_summaries
            try:
                build_raptor_summaries(
                    config, 
                    console,
                    chunk_store=chunk_store,
                    bm25=bm25,
                    vector_store=vector_store,
                    embedder=embedder
                )
            except Exception as exc:
                log.exception("RAPTOR indexing failed")
                if console:
                    console.print(f"[red]Summary generation error:[/red] {exc}")

    finally:
        # Release QdrantLocal file locks (important for Windows UX-03 safety)
        if hasattr(vector_store, "close"):
            try:
                vector_store.close()
            except Exception:
                pass

        # T1 memory policy: unload embedder before LLM is loaded
        model_manager.after_ingestion(config)

    return IngestResult(
        files_processed=files_processed,
        chunks_added=chunks_added,
        files_skipped=files_skipped,
        errors=errors,
    )


def remove_document(
    file_path: Path, 
    config: RAGConfig,
    chunk_store=None,
    bm25=None,
    vector_store=None,
    tracker=None
) -> int:
    """
    Remove a document and all its chunks from vector store, BM25 index,
    and SQLite chunk store using StorageTransactionManager.
    """
    from rag.retrieval.bm25_index import BM25Index
    from rag.retrieval.vector_store import VectorStore
    from rag.storage.chunk_store import ChunkStore
    from rag.storage.ingestion_tracker import IngestionTracker
    from rag.storage.transaction_manager import StorageTransactionManager

    owns_vector_store = vector_store is None
    
    chunk_store = chunk_store or ChunkStore(config)
    bm25 = bm25 or BM25Index(config)
    vector_store = vector_store or VectorStore(config)
    tracker = tracker or IngestionTracker(config)

    tx_manager = StorageTransactionManager(config)
    n = tx_manager.execute_remove(
        file_path=file_path,
        chunk_store=chunk_store,
        tracker=tracker,
        bm25=bm25,
        vector_store=vector_store,
    )
    
    if owns_vector_store and hasattr(vector_store, "close"):
        vector_store.close()

    log.info("Removed document %s (%d chunks).", file_path.name, n)
    return n


def sync_directory(
    directory: Path,
    config: RAGConfig,
    recursive: bool = True,
    console: Console | None = None,
) -> SyncResult:
    """
    Synchronise a directory with the knowledge base:
      1. Collect files currently on disk.
      2. Fetch files indexed under directory from IngestionTracker.
      3. Compute diff:
         - New files  → ingest_path()
         - Removed    → remove_document()
         - Modified   → re-ingest (remove then ingest)
         - Relocated  → update SQLite in place (no re-embedding)

    Returns:
        SyncResult with counts of added, removed, re-indexed, and errors.
    """
    from rag.storage.ingestion_tracker import IngestionTracker, compute_file_hash
    tracker = IngestionTracker(config)

    disk_files = {f.resolve(): f for f in _collect_files(directory, recursive)}
    disk_paths = set(disk_files.keys())

    indexed = {Path(r["filepath"]): r["content_hash"] for r in tracker.list_all()}
    indexed_paths = set(indexed.keys())

    to_add = disk_paths - indexed_paths
    to_remove = indexed_paths - disk_paths

    common = disk_paths & indexed_paths
    to_reindex: list[Path] = []
    for p in common:
        try:
            current_hash = compute_file_hash(disk_files[p])
            if current_hash != indexed.get(p):
                to_reindex.append(p)
        except OSError:
            pass

    # ── Relocation Detection (CRIT-03) ──────────────────────────────
    from rag.storage.db_manager import DatabaseManager
    db = DatabaseManager.get_connection(config)

    # Pre-compute target file hashes once to avoid O(N*M) hash reads
    new_hashes: dict[Path, str] = {}
    for new_path in list(to_add):
        try:
            new_hashes[new_path] = compute_file_hash(disk_files[new_path])
        except OSError:
            pass

    relocated: set[tuple[Path, Path]] = set()
    to_remove_list = list(to_remove)
    for old_path in to_remove_list:
        old_hash = indexed.get(old_path)
        if not old_hash:
            continue
        to_add_list = list(to_add)
        for new_path in to_add_list:
            if new_path not in to_add:
                continue
            new_hash = new_hashes.get(new_path)
            if new_hash and new_hash == old_hash:
                # File moved or renamed — relocate in place!
                relocated.add((old_path, new_path))
                to_remove.discard(old_path)
                to_add.discard(new_path)

                old_str = str(old_path.resolve())
                new_str = str(new_path.resolve())
                new_name = new_path.name

                db.execute(
                    "UPDATE file_tracker SET filepath = ? WHERE filepath = ?",
                    (new_str, old_str),
                )
                db.execute(
                    "UPDATE chunks SET source = ?, filename = ? WHERE source = ?",
                    (new_str, new_name, old_str),
                )
                db.commit()
                log.info("Relocated document %s -> %s", old_path.name, new_path.name)
                if console:
                    console.print(f"  [cyan]relocated[/cyan] {old_path.name} → {new_path.name}")
                break

    added = 0
    removed = 0
    reindexed = 0
    errors: list[str] = []

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
