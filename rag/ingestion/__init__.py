"""
rag/ingestion/__init__.py — Public API for the ingestion package.

Commands layer imports ONLY from here — never from internal ingestion modules.

    from rag.ingestion import ingest_path, remove_document, sync_directory

Phase 0: These functions are stubs. They define the interface (type signatures,
         docstrings, return types) that Phase 1 will implement.
Phase 1: Replace NotImplementedError with real implementation in this file,
         delegating to parsers/, chunker.py, deduplicator.py, and storage/.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag.types import IngestResult, SyncResult

if TYPE_CHECKING:
    from rich.console import Console
    from rag.config import RAGConfig


def ingest_path(
    path: Path,
    config: "RAGConfig",
    recursive: bool = False,
    console: "Console | None" = None,
) -> IngestResult:
    """
    Ingest all supported documents found at `path` into the knowledge base.

    Supported types: .pdf, .docx, .md, .txt, .png, .jpg, .mp3, .wav, .m4a

    Behaviour:
    - If `path` is a file: ingest that single file.
    - If `path` is a directory and recursive=False: ingest top-level files only.
    - If `path` is a directory and recursive=True: ingest all files in all subdirs.
    - Files already indexed with unchanged content hash are skipped (dedup).
    - Files with a changed hash are re-indexed (old chunks deleted first).

    Args:
        path:      Target file or directory (absolute, already validated by caller).
        config:    Loaded RAGConfig.
        recursive: If True, recurse into subdirectories.
        console:   Rich Console for progress output. If None, output is suppressed.

    Returns:
        IngestResult with counts of files processed, chunks added, files skipped.
    """
    raise NotImplementedError("ingest_path() — implemented in Phase 1")


def remove_document(path: Path, config: "RAGConfig") -> int:
    """
    Remove a document and all its indexed chunks from the knowledge base.

    The document file itself is NOT deleted from disk.

    Args:
        path:   Absolute path to the source document.
        config: Loaded RAGConfig.

    Returns:
        Number of chunks removed.
    """
    raise NotImplementedError("remove_document() — implemented in Phase 1")


def sync_directory(
    directory: Path,
    config: "RAGConfig",
    recursive: bool = False,
    console: "Console | None" = None,
) -> SyncResult:
    """
    Synchronise a directory with the knowledge base.

    - Files present in the directory but not in the index are ingested.
    - Files in the index that no longer exist on disk are removed.
    - Files whose content hash has changed since last ingestion are re-indexed.

    Args:
        directory: Target directory (absolute, already validated by caller).
        config:    Loaded RAGConfig.
        recursive: If True, recurse into subdirectories.
        console:   Rich Console for progress output.

    Returns:
        SyncResult with counts of added, removed, and re-indexed documents.
    """
    raise NotImplementedError("sync_directory() — implemented in Phase 2")
