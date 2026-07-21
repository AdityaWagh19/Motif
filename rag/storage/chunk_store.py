"""
rag/storage/chunk_store.py — SQLite-backed store for Chunk objects.

This is the authoritative record of every indexed chunk's text and metadata.
Qdrant holds the vectors; this store holds the full content.

Dependency graph position:
    chunk_store  →  rag.types  →  (stdlib only)

No other rag modules are imported here.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig

from rag.types import Chunk

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
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
    indexed_at   TEXT NOT NULL DEFAULT '',
    parent_id    TEXT
);
"""

_CREATE_INDEX_SOURCE = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);"
)
_CREATE_INDEX_SOURCE_TYPE = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);"
)
_CREATE_INDEX_CONTENT_HASH = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);"
)

_INSERT = """
INSERT OR REPLACE INTO chunks (
    id, text, source, filename, source_type,
    char_start, char_end, page, section,
    start_time, end_time,
    has_table, has_image, is_ocr, language,
    content_hash, token_count, indexed_at, parent_id
) VALUES (
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?
);
"""


def _chunk_to_row(chunk: Chunk) -> tuple:
    """Convert a Chunk to a tuple in the order expected by _INSERT."""
    return (
        chunk.id,
        chunk.text,
        chunk.source,
        chunk.filename,
        chunk.source_type,
        chunk.char_start,
        chunk.char_end,
        chunk.page,
        chunk.section,
        chunk.start_time,
        chunk.end_time,
        int(chunk.has_table),
        int(chunk.has_image),
        int(chunk.is_ocr),
        chunk.language,
        chunk.content_hash,
        chunk.token_count,
        chunk.indexed_at,
        chunk.parent_id,
    )


def _row_to_chunk(row: tuple) -> Chunk:
    """Convert a DB row (ordered by SELECT * columns) to a Chunk."""
    (
        id_, text, source, filename, source_type,
        char_start, char_end, page, section,
        start_time, end_time,
        has_table, has_image, is_ocr, language,
        content_hash, token_count, indexed_at, parent_id
    ) = row
    return Chunk(
        id=id_,
        text=text,
        source=source,
        filename=filename,
        source_type=source_type,
        char_start=char_start or 0,
        char_end=char_end or 0,
        page=page,
        section=section,
        start_time=start_time,
        end_time=end_time,
        has_table=bool(has_table),
        has_image=bool(has_image),
        is_ocr=bool(is_ocr),
        language=language,
        content_hash=content_hash or "",
        token_count=token_count or 0,
        indexed_at=indexed_at or "",
        parent_id=parent_id,
    )


# ---------------------------------------------------------------------------
# ChunkStore
# ---------------------------------------------------------------------------

class ChunkStore:
    """
    SQLite-backed persistent store for Chunk objects.

    WAL mode allows concurrent readers during a long ingestion write.
    All writes are synchronous — no background threads.

    Usage:
        store = ChunkStore(config)
        store.insert(chunk)
        chunk = store.fetch(chunk_id)
        store.close()
    """

    def __init__(self, config: RAGConfig) -> None:  # noqa: F821
        db_path: Path = config.db_root / "chunks.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        log.debug("Opening ChunkStore at %s", db_path)

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute(_CREATE_TABLE)
        
        # 7-B Migration: add parent_id column if it doesn't exist
        try:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN parent_id TEXT;")
        except sqlite3.OperationalError:
            pass  # Column already exists

        self._conn.execute(_CREATE_INDEX_SOURCE)
        self._conn.execute(_CREATE_INDEX_SOURCE_TYPE)
        self._conn.execute(_CREATE_INDEX_CONTENT_HASH)
        self._conn.commit()

    def __enter__(self) -> ChunkStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert(self, chunk: Chunk) -> None:
        """Insert or replace a single Chunk."""
        log.debug("ChunkStore.insert id=%s", chunk.id)
        self._conn.execute(_INSERT, _chunk_to_row(chunk))
        self._conn.commit()

    def insert_batch(self, chunks: list[Chunk]) -> None:
        """
        Insert a list of Chunks in a single transaction.
        Rolls back the entire transaction on any error.
        Silently no-ops for empty input.
        """
        if not chunks:
            return
        log.debug("ChunkStore.insert_batch count=%d", len(chunks))
        rows = [_chunk_to_row(c) for c in chunks]
        try:
            with self._conn:  # context manager: commit on success, rollback on error
                self._conn.executemany(_INSERT, rows)
        except sqlite3.Error:
            log.exception("ChunkStore.insert_batch failed — transaction rolled back")
            raise

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def fetch(self, chunk_id: str) -> Chunk | None:
        """Return the Chunk with the given id, or None if not found."""
        log.debug("ChunkStore.fetch id=%s", chunk_id)
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def fetch_parent(self, chunk: Chunk) -> Chunk | None:
        """
        7-B: Fetch the parent chunk of this child chunk.
        Returns None if chunk has no parent_id or parent doesn't exist.
        """
        if not chunk.parent_id:
            return None
        return self.fetch(chunk.parent_id)

    def fetch_batch(self, chunk_ids: list[str]) -> list[Chunk]:
        """
        Return all Chunks whose id is in chunk_ids.
        Order is not guaranteed. Missing ids are silently skipped.
        Returns [] for empty input.
        """
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def fetch_by_source(self, source: str) -> list[Chunk]:
        """Return all Chunks whose source equals the given path string."""
        log.debug("ChunkStore.fetch_by_source source=%s", source)
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE source = ?", (source,)
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks whose source equals the given path string.
        Returns the number of rows deleted.
        """
        log.debug("ChunkStore.delete_by_source source=%s", source)
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE source = ?", (source,)
        )
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of chunks in the store."""
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def count_documents(self) -> int:
        """Return number of distinct source documents."""
        return self._conn.execute(
            "SELECT COUNT(DISTINCT source) FROM chunks"
        ).fetchone()[0]

    def list_ids(self) -> list[str]:
        """Return a list of all chunk IDs in the store."""
        rows = self._conn.execute("SELECT id FROM chunks").fetchall()
        return [r[0] for r in rows]

    def list_sources(self) -> list[str]:
        """Return sorted list of distinct source paths. Used by /status and /sync."""
        rows = self._conn.execute(
            "SELECT DISTINCT source FROM chunks ORDER BY source"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
