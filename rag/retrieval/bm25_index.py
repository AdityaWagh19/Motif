"""
rag/retrieval/bm25_index.py — BM25 lexical index over chunk text.

Complements dense vector search. Persisted to disk as a pickle file so it
survives process restarts without reindexing.

Backend: rank_bm25.BM25Okapi
Tokenization: text.lower().split()  — consistent, reproducible, zero-dependency.

Persistence:  config.db_root / "bm25" / "index.pkl"
Pickle format: {"corpus_tokens": List[List[str]], "chunk_ids": List[str], "version": 1}

Phase 4 adds a tantivy backend that auto-activates at 100K chunks.
The public API (add, search, delete, save) is identical across backends.

Dependency graph position:
    bm25_index  →  rank_bm25  (third-party, no rag internals)
    bm25_index  →  rag.types  →  (stdlib only)
"""
from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from rag.types import Chunk

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

_PICKLE_VERSION = 1

# Auto-migrate to tantivy when corpus exceeds this many chunks.
TANTIVY_THRESHOLD = 100_000


class BM25Index:
    """
    In-memory BM25 index backed by rank_bm25.BM25Okapi.

    The index is persisted to a single pickle file. On startup, if the file
    exists it is loaded automatically. After each mutating operation the
    in-memory BM25 object is rebuilt (O(n) but fast for < 100K chunks).

    Thread-safety: Not thread-safe. Motif is single-threaded — no locking needed.

    Typical usage:
        index = BM25Index(config)
        index.add_batch(chunks)    # add during ingestion
        results = index.search("query", top_k=20)  # at query time
    """

    def __init__(self, config: RAGConfig) -> None:  # noqa: F821
        self._index_path: Path = config.db_root / "bm25" / "index.pkl"
        self._tantivy_path: Path = config.db_root / "tantivy_index"
        self._corpus_tokens: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._dirty: bool = False
        self._backend: Literal["rank_bm25", "tantivy"] = "rank_bm25"
        self._tantivy_index: object = None  # tantivy.Index when active

        if self._index_path.exists():
            self._load()

        # Auto-migrate if corpus is already large (e.g. after restart)
        self._check_and_maybe_migrate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load index from disk. On corruption, start fresh with a warning."""
        try:
            with open(str(self._index_path), "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or data.get("version") != _PICKLE_VERSION:
                raise ValueError("Incompatible BM25 index version — rebuilding.")
            self._corpus_tokens = data["corpus_tokens"]
            self._chunk_ids = data["chunk_ids"]
            log.debug(
                "BM25Index loaded from disk: %d chunks", len(self._chunk_ids)
            )
            self._rebuild_bm25()
        except Exception as exc:
            log.warning("BM25Index: failed to load index (%s) — starting fresh.", exc)
            self._corpus_tokens = []
            self._chunk_ids = []
            self._bm25 = None
            self._dirty = False

    def _rebuild_bm25(self) -> None:
        """
        Rebuild the BM25Okapi object from the current corpus.

        Intentionally does NOT touch self._dirty — dirty state is the caller's
        responsibility. Only save() clears the dirty flag once data is on disk.
        """
        if not self._corpus_tokens:
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(self._corpus_tokens)
            chunk_count = len(self._chunk_ids)
            if chunk_count > 5_000:
                log.warning(
                    "BM25 index has %d chunks. Tantivy backend will auto-activate at %d chunks.",
                    chunk_count, TANTIVY_THRESHOLD,
                )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenise text: lowercase, whitespace split. Consistent with query tokenization."""
        return text.lower().split()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Persist the current index to disk.

        Uses an atomic write (temp file + rename) to prevent corruption
        if the process dies mid-write.

        No-op if the index has not changed since the last save.
        """
        if not self._dirty:
            return

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus_tokens": self._corpus_tokens,
            "chunk_ids": self._chunk_ids,
            "version": _PICKLE_VERSION,
        }
        # Atomic write: write to temp, then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._index_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, str(self._index_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self._dirty = False
        log.debug("BM25Index saved: %d chunks", len(self._chunk_ids))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunk: Chunk) -> None:
        """
        Add a single chunk to the index.

        If a chunk with the same id already exists, it is replaced (delete + add).
        Rebuilds the BM25Okapi object immediately.
        """
        if chunk.id in self._chunk_ids:
            self.delete(chunk.id, _rebuild=False)

        tokens = self._tokenize(chunk.text)
        self._corpus_tokens.append(tokens)
        self._chunk_ids.append(chunk.id)
        self._dirty = True
        self._rebuild_bm25()

    def add_batch(self, chunks: list[Chunk], save: bool = True) -> None:
        """
        Add a list of chunks to the index in one operation.

        Handles duplicate ids (replaces existing entries).
        Performs a single BM25Okapi rebuild at the end.
        Saves to disk if save=True.
        """
        if not chunks:
            return

        # Handle duplicates first (collect all ids to replace)
        incoming_ids = {c.id for c in chunks}
        ids_to_delete = incoming_ids & set(self._chunk_ids)
        if ids_to_delete:
            # Remove all existing entries for these ids
            keep_mask = [cid not in ids_to_delete for cid in self._chunk_ids]
            self._corpus_tokens = [
                t for t, keep in zip(self._corpus_tokens, keep_mask) if keep
            ]
            self._chunk_ids = [
                cid for cid, keep in zip(self._chunk_ids, keep_mask) if keep
            ]

        for chunk in chunks:
            self._corpus_tokens.append(self._tokenize(chunk.text))
            self._chunk_ids.append(chunk.id)

        self._dirty = True
        self._rebuild_bm25()
        if save:
            self.save()

    def delete(self, chunk_id: str, _rebuild: bool = True) -> bool:
        """
        Remove a single chunk from the index by its id.

        Returns True if found and removed, False if not found.
        Rebuilds the BM25Okapi object if _rebuild=True.
        """
        try:
            idx = self._chunk_ids.index(chunk_id)
        except ValueError:
            return False

        del self._corpus_tokens[idx]
        del self._chunk_ids[idx]
        self._dirty = True
        if _rebuild:
            self._rebuild_bm25()
        return True

    def delete_by_source(self, source: str, chunk_ids: list[str]) -> int:
        """
        Remove all chunks with the given ids from the index.

        Args:
            source:    Not used for deletion logic — kept for logging.
            chunk_ids: The specific chunk ids to remove.

        Returns:
            Number of chunks actually removed.
        """
        ids_set = set(chunk_ids)
        original_count = len(self._chunk_ids)

        keep_mask = [cid not in ids_set for cid in self._chunk_ids]
        self._corpus_tokens = [
            t for t, keep in zip(self._corpus_tokens, keep_mask) if keep
        ]
        self._chunk_ids = [
            cid for cid, keep in zip(self._chunk_ids, keep_mask) if keep
        ]

        removed = original_count - len(self._chunk_ids)
        if removed > 0:
            self._dirty = True
            self._rebuild_bm25()
            log.debug(
                "BM25Index.delete_by_source source=%s removed=%d", source, removed
            )
        return removed

    def rebuild(self) -> None:
        """Force a full BM25Okapi rebuild and save to disk."""
        self._dirty = True
        self._rebuild_bm25()
        self.save()
        self._check_and_maybe_migrate()

    # ------------------------------------------------------------------
    # Tantivy backend
    # ------------------------------------------------------------------

    def _check_and_maybe_migrate(self) -> None:
        """Migrate from rank_bm25 to tantivy if corpus exceeds the threshold."""
        if self._backend == "rank_bm25" and len(self._chunk_ids) >= TANTIVY_THRESHOLD:
            self._migrate_to_tantivy()

    def _migrate_to_tantivy(self) -> None:
        """
        Build a tantivy index from the current corpus.

        After migration, all search() calls use tantivy. The rank_bm25 pickle
        is kept on disk as backup but is no longer used for queries.
        """
        try:
            import tantivy  # type: ignore[import]
        except ImportError:
            log.warning(
                "Corpus >= %d chunks but tantivy is not installed. "
                "Install with: pip install tantivy. Staying on rank_bm25.",
                TANTIVY_THRESHOLD,
            )
            return

        log.info(
            "Migrating BM25 backend to tantivy (%d chunks)...", len(self._chunk_ids)
        )
        self._tantivy_path.mkdir(parents=True, exist_ok=True)  # type: ignore[call-arg]

        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("text", stored=True)
        schema_builder.add_text_field("chunk_id", stored=True)
        schema = schema_builder.build()

        index = tantivy.Index(schema, path=str(self._tantivy_path))
        writer = index.writer(heap_size=50_000_000)

        for tokens, chunk_id in zip(self._corpus_tokens, self._chunk_ids):
            writer.add_document(tantivy.Document(
                text=" ".join(tokens),
                chunk_id=chunk_id,
            ))
        writer.commit()
        index.reload()

        self._tantivy_index = index
        self._backend = "tantivy"
        log.info("BM25 migration to tantivy complete.")

    def _search_tantivy(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search using the tantivy in-memory index."""
        try:
            searcher = self._tantivy_index.searcher()  # type: ignore[union-attr]
            query_obj = self._tantivy_index.parse_query(query, ["text"])  # type: ignore[union-attr]
            results = searcher.search(query_obj, top_k).hits
            return [
                (searcher.doc(addr)["chunk_id"][0], float(score))
                for score, addr in results
            ]
        except Exception as e:
            log.warning("tantivy search failed (%s) — falling back to rank_bm25.", e)
            return self._search_rank_bm25(query, top_k)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Search the index for the top-k most relevant chunks.

        Automatically dispatches to tantivy backend when corpus >= 100K chunks.

        Args:
            query:  The search string. Tokenised the same way as at index time.
            top_k:  Maximum number of results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending.
            Zero-score results are excluded (no term overlap with query).
            Returns [] if the index is empty.
        """
        if self._backend == "tantivy":
            return self._search_tantivy(query, top_k)
        return self._search_rank_bm25(query, top_k)

    def _search_rank_bm25(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """rank_bm25 backend search implementation."""
        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)  # numpy array, length = corpus size

        # Get top_k indices by score (unsorted first, then sort)
        n = len(scores)
        k = min(top_k, n)

        # Partial sort: argpartition gives k-largest indices without full sort.
        # When k == n, skip argpartition (would error with k == 0 edge case).
        if k == n:
            top_indices = list(range(n))
        else:
            top_indices = list(np.argpartition(scores, -k)[-k:])  # type: ignore[arg-type]

        # Filter: exclude score == 0.0 (query tokens are entirely OOV — no lexical
        # overlap at all). Do NOT filter negative scores: BM25 IDF is legitimately
        # negative when df > N/2 (e.g., in corpora with < 3 documents), but the
        # scores are still meaningful for relative ranking.
        results = [
            (self._chunk_ids[i], float(scores[i]))
            for i in top_indices
            if float(scores[i]) != 0.0
        ]

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of chunks currently in the index."""
        return len(self._chunk_ids)

    @classmethod
    def count_from_disk(cls, config: RAGConfig) -> int:
        """
        Return the number of chunks by reading the index from disk.
        Does not load the BM25Okapi object into memory.
        """
        index_path = config.db_root / "bm25" / "index.pkl"
        if not index_path.exists():
            return 0
        try:
            with open(str(index_path), "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return len(data.get("chunk_ids", []))
            return 0
        except Exception:
            return 0
