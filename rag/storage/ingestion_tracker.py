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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS files (
    filepath     TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at   TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0
);
"""

_UPSERT = """
INSERT OR REPLACE INTO files (filepath, content_hash, indexed_at, chunk_count)
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

    Usage:
        tracker = IngestionTracker(config)
        if not tracker.is_indexed(path) or tracker.get_hash(path) != new_hash:
            ingest(path)
            tracker.update(path, new_hash, chunk_count)
    """

    def __init__(self, config: "RAGConfig") -> None:  # noqa: F821
        db_path: Path = config.db_root / "ingestion_tracker.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        log.debug("Opening IngestionTracker at %s", db_path)

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_indexed(self, path: Path) -> bool:
        """Return True if this path has been indexed at least once."""
        key = str(path.resolve())
        row = self._conn.execute(
            "SELECT 1 FROM files WHERE filepath = ?", (key,)
        ).fetchone()
        return row is not None

    def get_hash(self, path: Path) -> Optional[str]:
        """
        Return the content hash recorded for this path, or None if not tracked.
        """
        key = str(path.resolve())
        row = self._conn.execute(
            "SELECT content_hash FROM files WHERE filepath = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def list_all(self) -> List[Dict]:
        """
        Return all tracked file records as a list of dicts.

        Each dict has keys: filepath, content_hash, indexed_at, chunk_count.
        Used by sync_directory() to compute the diff against the filesystem.
        """
        rows = self._conn.execute(
            "SELECT filepath, content_hash, indexed_at, chunk_count FROM files"
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
        now = datetime.now(timezone.utc).isoformat()
        log.debug("IngestionTracker.update filepath=%s hash=%s", key, content_hash)
        self._conn.execute(_UPSERT, (key, content_hash, now, chunk_count))
        self._conn.commit()

    def remove(self, path: Path) -> None:
        """Remove the tracking record for a given path. No-op if not tracked."""
        key = str(path.resolve())
        log.debug("IngestionTracker.remove filepath=%s", key)
        self._conn.execute("DELETE FROM files WHERE filepath = ?", (key,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
