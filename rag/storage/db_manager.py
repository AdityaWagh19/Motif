"""
rag/storage/db_manager.py — Centralized SQLite Connection & Schema Manager.

Consolidates all relational data for a workspace into a single SQLite database:
`motif_store.db`.

Tables managed:
  - workspace_meta : Key-value metadata and corpus_version UUID.
  - chunks         : Authoritative chunk text, metadata, and parent/child hierarchy.
  - file_tracker   : Ingestion file path to SHA-256 hash tracking.
  - query_cache    : LRU query results cache with corpus_version matching.
  - session_turns  : Persisted conversation history exchanges per session.
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

DB_FILENAME = "motif_store.db"

_CREATE_SCHEMA = """
-- 1. Workspace Metadata & Versioning
CREATE TABLE IF NOT EXISTS workspace_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 2. Authoritative Chunk Store
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
    parent_id    TEXT,
    FOREIGN KEY(parent_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);

-- 3. Ingestion File Tracking (Support for Content-Hash Relocation Detection)
CREATE TABLE IF NOT EXISTS file_tracker (
    filepath     TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at   TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_files_hash ON file_tracker(content_hash);

-- 4. LRU Query Result Cache (with Scoped Invalidation Support)
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key      TEXT PRIMARY KEY,
    corpus_version TEXT NOT NULL,
    file_filter    TEXT,
    query_text     TEXT NOT NULL,
    answer_text    TEXT NOT NULL,
    citations      TEXT NOT NULL,
    passages_used  INTEGER NOT NULL,
    latency_ms     REAL,
    accessed_at    REAL NOT NULL,
    created_at     REAL NOT NULL,
    used_sources   TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_qc_accessed ON query_cache(accessed_at);
CREATE INDEX IF NOT EXISTS idx_qc_version ON query_cache(corpus_version);
CREATE INDEX IF NOT EXISTS idx_qc_file_filter ON query_cache(file_filter);

-- 5. Session History Storage
CREATE TABLE IF NOT EXISTS session_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON session_turns(session_id, turn_index);
"""


class DatabaseManager:
    """
    Centralized connection pool and schema manager for motif_store.db.
    """

    _connections: dict[str, sqlite3.Connection] = {}

    @classmethod
    def get_connection(cls, config: RAGConfig) -> sqlite3.Connection:
        """Return an open SQLite connection to motif_store.db for the given workspace."""
        db_path = config.db_root / DB_FILENAME
        db_key = str(db_path.resolve())

        conn = cls._connections.get(db_key)
        if conn is not None:
            try:
                conn.execute("SELECT 1;")
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                conn = None
                cls._connections.pop(db_key, None)

        if conn is None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(_CREATE_SCHEMA)
            cls._ensure_corpus_version(conn)
            cls._connections[db_key] = conn
            log.debug("Opened DatabaseManager connection to %s", db_path)

        return conn

    @classmethod
    def _ensure_corpus_version(cls, conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT value FROM workspace_meta WHERE key = 'corpus_version'"
        ).fetchone()
        if row:
            return row[0]

        version = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO workspace_meta (key, value) VALUES ('corpus_version', ?)",
            (version,),
        )
        conn.commit()
        return version

    @classmethod
    def get_corpus_version(cls, config: RAGConfig) -> str:
        conn = cls.get_connection(config)
        return cls._ensure_corpus_version(conn)

    @classmethod
    def bump_corpus_version(cls, config: RAGConfig) -> str:
        conn = cls.get_connection(config)
        new_version = str(uuid.uuid4())
        conn.execute(
            "INSERT OR REPLACE INTO workspace_meta (key, value) VALUES ('corpus_version', ?)",
            (new_version,),
        )
        conn.commit()
        log.debug("Bumped corpus_version to %s", new_version)
        return new_version

    @classmethod
    def close_connection(cls, config: RAGConfig) -> None:
        db_path = config.db_root / DB_FILENAME
        db_key = str(db_path.resolve())
        conn = cls._connections.pop(db_key, None)
        if conn:
            try:
                conn.close()
                log.debug("Closed DatabaseManager connection to %s", db_path)
            except Exception as e:
                log.warning("Error closing database connection %s: %s", db_path, e)

    @classmethod
    def close_all(cls) -> None:
        for db_key, conn in list(cls._connections.items()):
            try:
                conn.close()
                log.debug("Closed connection %s", db_key)
            except Exception as e:
                log.warning("Error closing database %s: %s", db_key, e)
        cls._connections.clear()
