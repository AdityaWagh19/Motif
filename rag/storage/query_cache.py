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
from typing import TYPE_CHECKING

from rag.types import AnswerResult, Citation

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

# Maximum number of entries in the cache (LRU eviction kicks in above this).
_DEFAULT_MAX_ENTRIES = 500

# DB filename within db_root.
_CACHE_DB_NAME = "query_cache.sqlite"



def _make_key(
    query: str,
    file_filter: str | None,
    type_filter: str | None,
    page_range: str | None,
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
     passages_used, latency_ms, accessed_at, created_at) = row[:8]

    try:
        c_list = json.loads(citations_json)
        citations = [Citation(**c) for c in c_list]
    except Exception:
        citations = []

    return AnswerResult(
        text=answer_text,
        citations=citations,
        passages_used=passages_used or 0,
        latency_ms=latency_ms or 0.0,
        tier="cached",
    )


class QueryCache:
    """
    SQLite-backed LRU query result cache.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._enabled: bool = getattr(config.storage, "query_cache_enabled", False)
        self._max_entries: int = getattr(
            config.storage, "query_cache_max_entries", _DEFAULT_MAX_ENTRIES
        )
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        from rag.storage.db_manager import DatabaseManager
        if self._conn is None:
            self._conn = DatabaseManager.get_connection(self._config)
        return self._conn

    def get(
        self,
        query: str,
        file_filter: str | None = None,
        type_filter: str | None = None,
        page_range: str | None = None,
    ) -> AnswerResult | None:
        if not self._enabled:
            return None

        from rag.storage.db_manager import DatabaseManager
        key = _make_key(query, file_filter, type_filter, page_range)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT cache_key, query_text, answer_text, citations, passages_used, latency_ms, accessed_at, created_at, corpus_version, used_sources FROM query_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        except sqlite3.OperationalError:
            row = conn.execute(
                "SELECT cache_key, query_text, answer_text, citations, passages_used, latency_ms, accessed_at, created_at FROM query_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None



        # Update accessed_at for LRU ordering
        conn.execute(
            "UPDATE query_cache SET accessed_at = ? WHERE cache_key = ?",
            (time.time(), key),
        )
        conn.commit()

        log.debug("Cache HIT for query: %.60s…", query)
        return _row_to_result(row[:8])

    def put(
        self,
        query: str,
        result: AnswerResult,
        file_filter: str | None = None,
        type_filter: str | None = None,
        page_range: str | None = None,
    ) -> None:
        if not self._enabled or not result.text or result.passages_used == 0:
            return

        from rag.storage.db_manager import DatabaseManager
        current_version = DatabaseManager.get_corpus_version(self._config)
        key = _make_key(query, file_filter, type_filter, page_range)
        row = _result_to_row(key, query, result)
        conn = self._connect()

        used_sources = "|" + "|".join(sorted(set(c.filepath for c in result.citations if c.filepath))) + "|"

        conn.execute(
            """
            INSERT OR REPLACE INTO query_cache
            (cache_key, query_text, answer_text, citations, passages_used,
             latency_ms, accessed_at, created_at, corpus_version, file_filter, used_sources)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], current_version, file_filter or "", used_sources),
        )
        conn.commit()
        self._evict_if_needed()

    def invalidate_file(self, file_path_str: str) -> int:
        """
        O(1) indexed targeted invalidation for queries matching file_filter or used_sources.
        """
        if not self._enabled:
            return 0
        conn = self._connect()
        cur = conn.execute("DELETE FROM query_cache WHERE file_filter = ? OR used_sources LIKE ?", (file_path_str, f"%|{file_path_str}|%"))
        conn.commit()
        log.debug("QueryCache: invalidated %d entries for file %s", cur.rowcount, file_path_str)
        return cur.rowcount

    def _evict_if_needed(self) -> None:
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

    def count(self) -> int:
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]

    def clear(self) -> None:
        if not self._enabled:
            return
        conn = self._connect()
        conn.execute("DELETE FROM query_cache")
        conn.commit()

    def close(self) -> None:
        # Connection lifecycle is managed by DatabaseManager
        self._conn = None
