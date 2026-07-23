"""
rag/storage/ingestion_tracker.py — File hash tracking for the ingestion pipeline.

Records which files have been ingested and with what content hash.
Used by ingest_path() to skip unchanged files and by sync_directory() to
detect added, removed, and changed files.

compute_file_hash() is a module-level function — Phase 2 imports it directly.

Dependency graph position:
    ingestion_tracker  →  (stdlib only)

No rag modules are imported here.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helper (imported by Phase 2 ingestion pipeline)
# ---------------------------------------------------------------------------

def compute_file_hash(path: Path) -> str:
    """
    Compute the SHA-256 hash of a file's contents.

    Reads in 64 KB chunks to handle large files without loading them fully
    into memory.

    Args:
        path: Absolute path to the file.

    Returns:
        64-character lowercase hex digest string.

    Raises:
        FileNotFoundError: If path does not exist.
        PermissionError:   If the file cannot be read.
    """
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS file_tracker (
    filepath     TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at   TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0
);
"""

_UPSERT = """
INSERT OR REPLACE INTO file_tracker (filepath, content_hash, indexed_at, chunk_count)
VALUES (?, ?, ?, ?);
"""


# ---------------------------------------------------------------------------
# IngestionTracker
# ---------------------------------------------------------------------------

class IngestionTracker:
    """
    SQLite-backed tracker for ingested file paths and their content hashes.

    Each record stores:
        filepath     — absolute path (str) as primary key
        content_hash — SHA-256 hex digest of the file at ingestion time
        indexed_at   — ISO 8601 UTC timestamp
        chunk_count  — number of chunks produced from this file
    """

    def __init__(self, config: RAGConfig) -> None:  # noqa: F821
        from rag.storage.db_manager import DatabaseManager
        self._config = config
        self._conn = DatabaseManager.get_connection(config)

    def __enter__(self) -> IngestionTracker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_indexed(self, path: Path) -> bool:
        """Return True if this path has been indexed at least once."""
        key = str(path.resolve())
        row = self._conn.execute(
            "SELECT 1 FROM file_tracker WHERE filepath = ?", (key,)
        ).fetchone()
        return row is not None

    def get_hash(self, path: Path) -> str | None:
        """
        Return the content hash recorded for this path, or None if not tracked.
        """
        key = str(path.resolve())
        row = self._conn.execute(
            "SELECT content_hash FROM file_tracker WHERE filepath = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def list_all(self) -> list[dict]:
        """
        Return all tracked file records as a list of dicts.

        Each dict has keys: filepath, content_hash, indexed_at, chunk_count.
        Used by sync_directory() to compute the diff against the filesystem.
        """
        rows = self._conn.execute(
            "SELECT filepath, content_hash, indexed_at, chunk_count FROM file_tracker"
        ).fetchall()
        return [
            {
                "filepath": r[0],
                "content_hash": r[1],
                "indexed_at": r[2],
                "chunk_count": r[3],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def update(self, path: Path, content_hash: str, chunk_count: int) -> None:
        """
        Record (or update) an ingested file entry.

        Always stores the absolute, resolved path.
        Sets indexed_at to the current UTC time.
        """
        key = str(path.resolve())
        now = datetime.now(UTC).isoformat()
        log.debug("IngestionTracker.update filepath=%s hash=%s", key, content_hash)
        self._conn.execute(_UPSERT, (key, content_hash, now, chunk_count))
        self._conn.commit()

    def remove(self, path: Path) -> None:
        """Remove the tracking record for a given path. No-op if not tracked."""
        key = str(path.resolve())
        log.debug("IngestionTracker.remove filepath=%s", key)
        self._conn.execute("DELETE FROM file_tracker WHERE filepath = ?", (key,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close/release this tracker instance reference (lifecycle managed by DatabaseManager)."""
        self._conn = None
