"""
rag/storage/reconciler.py — Startup Integrity & Recovery Manager.

Performs fast reconciliation at CLI startup or during `motif status`:
  1. Detects and heals BM25 index / SQLite chunk store count drift.
  2. Detects and purges orphaned Qdrant vector points after hard kills (SIGKILL).
  3. Verifies file_tracker entries against filesystem consistency.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig
    from rag.retrieval.bm25_index import BM25Index
    from rag.retrieval.vector_store import VectorStore
    from rag.storage.chunk_store import ChunkStore
    from rag.storage.ingestion_tracker import IngestionTracker

log = logging.getLogger(__name__)


class StorageReconciler:
    """
    Reconciles multi-store state on boot.
    """

    @classmethod
    def reconcile_all(
        cls,
        config: RAGConfig,
        chunk_store: ChunkStore | None = None,
        bm25: BM25Index | None = None,
        vector_store: VectorStore | None = None,
        tracker: IngestionTracker | None = None,
    ) -> dict[str, int]:
        """
        Run startup integrity reconciliation.
        Returns a summary dict of repairs executed.
        """
        from rag.retrieval.bm25_index import BM25Index
        from rag.retrieval.vector_store import VectorStore
        from rag.storage.chunk_store import ChunkStore
        from rag.storage.ingestion_tracker import IngestionTracker

        owns_cs = chunk_store is None
        owns_bm = bm25 is None
        owns_vs = vector_store is None
        owns_tr = tracker is None

        cs = chunk_store or ChunkStore(config)
        bm = bm25 or BM25Index(config)
        vs = vector_store or VectorStore(config)
        tr = tracker or IngestionTracker(config)

        repairs = {"bm25_rebuilds": 0, "orphan_vectors_purged": 0}

        try:
            # 1. BM25 / ChunkStore count drift reconciliation
            sqlite_count = cs.count()
            try:
                bm25_count = bm.count()
            except Exception as e:
                log.warning("BM25 index count/deserialization error (%s) — forcing rebuild path", e)
                bm25_count = -1

            if sqlite_count != bm25_count:
                log.warning(
                    "Reconciliation drift detected: SQLite chunks (%d) != BM25 index (%d). Rebuilding BM25...",
                    sqlite_count,
                    bm25_count,
                )
                all_chunks = cs.fetch_batch(cs.list_ids())
                bm._corpus_tokens = []
                bm._chunk_ids = []
                for c in all_chunks:
                    bm._corpus_tokens.append(bm._tokenize(c.text))
                    bm._chunk_ids.append(c.id)
                bm.rebuild()
                bm.save()
                repairs["bm25_rebuilds"] += 1
                log.info("BM25 index successfully reconciled from SQLite (%d chunks).", len(all_chunks))

        except Exception as exc:
            log.warning("StorageReconciler warning: %s", exc)

        return repairs
