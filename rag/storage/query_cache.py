"""
rag/storage/query_cache.py — SQLite-backed LRU query result cache.

Phase 4: Avoids re-running the full retrieval + rerank + LLM pipeline for
identical or semantically very similar queries.

Design:
  - Cache key: SHA-256 hash of (query_text + file_filter + type_filter + page_range)
  - Storage: SQLite table `query_cache` in the db_root directory
  - Eviction: LRU — when the cache reaches `max_entries`, the oldest-accessed
    entry is evicted. Access time is updated on every cache hit.
  - Max entries: 500 (configurable via config or default)
  - Cache validity: no TTL — entries are valid until evicted by LRU.
  - Stored: serialised AnswerResult JSON (text + citation dicts)

Cache hit = skip query expansion, retrieval, reranking, and LLM generation.
First-token latency improvement: from seconds to milliseconds on T1.

Privacy note: If `storage.query_cache_enabled = false` (default), the cache
is completely disabled. A yellow warning is printed on startup when enabled.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from rag.types import AnswerResult, Citation

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

# Maximum number of entries in the cache (LRU eviction kicks in above this).
_DEFAULT_MAX_ENTRIES = 500

# DB filename within db_root.
_CACHE_DB_NAME = "query_cache.sqlite"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key   TEXT PRIMARY KEY,
    query_text  TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    citations   TEXT NOT NULL,   -- JSON array of citation dicts
    passages_used INTEGER NOT NULL,
    latency_ms  REAL,
    accessed_at REAL NOT NULL,   -- Unix timestamp of last access (for LRU)
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qc_accessed ON query_cache (accessed_at);
"""


def _make_key(
    query: str,
    file_filter: Optional[str],
    type_filter: Optional[str],
    page_range: Optional[str],
) -> str:
    """Compute a stable SHA-256 cache key from the query + filter parameters."""
    parts = [
        query.strip().lower(),
        file_filter or "",
        type_filter or "",
        page_range or "",
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _result_to_row(key: str, query: str, result: AnswerResult) -> tuple:
    """Serialise an AnswerResult to a SQLite row tuple."""
    citations_json = json.dumps([
        {
            "number": c.number,
            "source_type": c.source_type,
            "filepath": c.filepath,
            "filename": c.filename,
            "page": c.page,
            "section": c.section,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "relevance_score": c.relevance_score,
            "excerpt": c.excerpt,
        }
        for c in result.citations
    ])
    now = time.time()
    return (
        key,
        query,
        result.text,
        citations_json,
        result.passages_used,
        result.latency_ms,
        now,
        now,
    )


def _row_to_result(row: tuple) -> AnswerResult:
    """Deserialise a SQLite row to an AnswerResult."""
    (key, query_text, answer_text, citations_json,
     passages_used, latency_ms, accessed_at, created_at) = row

    citations_data = json.loads(citations_json)
    citations = [
        Citation(
            number=c["number"],
            source_type=c["source_type"],
            filepath=c["filepath"],
            filename=c["filename"],
            page=c.get("page"),
            section=c.get("section"),
            start_time=c.get("start_time", 0.0),
            end_time=c.get("end_time", 0.0),
            relevance_score=c.get("relevance_score", 0.0),
            excerpt=c.get("excerpt", ""),
        )
        for c in citations_data
    ]
    return AnswerResult(
        text=answer_text,
        citations=citations,
        passages_used=passages_used or 0,
        latency_ms=latency_ms,
        tier="cached",
    )


class QueryCache:
    """
    SQLite-backed LRU cache for query results.

    Thread-safety: single-writer SQLite (WAL mode) — safe for the single-user
    REPL use case.
    """

    def __init__(self, config: "RAGConfig", max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        """
        Open (or create) the cache database.

        Args:
            config:      RAGConfig — reads storage.db_path.
            max_entries: Maximum cache size before LRU eviction.
        """
        import os
        db_root = config.db_root
        self._db_path = db_root / _CACHE_DB_NAME
        self._max_entries = max_entries
        self._conn: Optional[sqlite3.Connection] = None
        self._enabled: bool = getattr(config.storage, "query_cache_enabled", False)
        if self._enabled:
            os.makedirs(str(db_root), exist_ok=True)
            self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._connect()
        conn.executescript(_CREATE_TABLE)
        conn.commit()

    def get(
        self,
        query: str,
        file_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        page_range: Optional[str] = None,
    ) -> Optional[AnswerResult]:
        """
        Retrieve a cached result, or None on cache miss.

        Updates accessed_at on hit (LRU maintenance).

        Args:
            query:       The user's query.
            file_filter: File filter modifier (or None).
            type_filter: Type filter modifier (or None).
            page_range:  Page range filter modifier (or None).

        Returns:
            AnswerResult with tier="cached" on hit, None on miss.
        """
        if not self._enabled:
            return None
        key = _make_key(query, file_filter, type_filter, page_range)
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM query_cache WHERE cache_key = ?", (key,)
        ).fetchone()

        if row is None:
            return None

        # Update accessed_at for LRU ordering.
        conn.execute(
            "UPDATE query_cache SET accessed_at = ? WHERE cache_key = ?",
            (time.time(), key),
        )
        conn.commit()

        log.debug("Cache HIT for query: %.60s…", query)
        return _row_to_result(row)

    def put(
        self,
        query: str,
        result: AnswerResult,
        file_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        page_range: Optional[str] = None,
    ) -> None:
        """
        Store a query result in the cache.

        Does not cache if result.text is empty or if the result indicates
        no passages were found.

        Args:
            query:       The user's query.
            result:      AnswerResult to store.
            file_filter: File filter modifier (or None).
            type_filter: Type filter modifier (or None).
            page_range:  Page range modifier (or None).
        """
        if not self._enabled:
            return
        if not result.text:
            return
        if result.passages_used == 0:
            return   # Don't cache "no results" — the index might change.

        key = _make_key(query, file_filter, type_filter, page_range)
        row = _result_to_row(key, query, result)
        conn = self._connect()

        conn.execute(
            """
            INSERT OR REPLACE INTO query_cache
            (cache_key, query_text, answer_text, citations, passages_used,
             latency_ms, accessed_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        conn.commit()

        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-used entries if cache exceeds max_entries."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        if count > self._max_entries:
            excess = count - self._max_entries
            conn.execute(
                """
                DELETE FROM query_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM query_cache
                    ORDER BY accessed_at ASC
                    LIMIT ?
                )
                """,
                (excess,),
            )
            conn.commit()
            log.debug("QueryCache: evicted %d LRU entries.", excess)

    def count(self) -> int:
        """Return the current number of entries in the cache."""
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]

    def clear(self) -> None:
        """Delete all cache entries."""
        conn = self._connect()
        conn.execute("DELETE FROM query_cache")
        conn.commit()
        log.info("QueryCache: cleared all entries.")

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
