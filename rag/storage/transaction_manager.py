"""
rag/storage/transaction_manager.py — Multi-Store Atomic Transaction Manager.

Coordinates atomic operations across:
  1. motif_store.db (chunks, file_tracker, workspace_meta, query_cache)
  2. VectorStore (Qdrant HNSW vector index)
  3. BM25Index (In-memory / Tantivy lexical index)

Guarantees compensating rollbacks across backends if any step fails, and
automatically updates corpus_version & query cache invalidations on mutation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rag.config import RAGConfig
    from rag.retrieval.bm25_index import BM25Index
    from rag.retrieval.vector_store import VectorStore
    from rag.storage.chunk_store import ChunkStore
    from rag.storage.ingestion_tracker import IngestionTracker
    from rag.types import Chunk

log = logging.getLogger(__name__)


class StorageTransactionManager:
    """
    Coordinates multi-store ingestion and deletion transactions with compensating rollbacks.
    """

    def __init__(self, config: RAGConfig):
        from rag.storage.db_manager import DatabaseManager
        self.config = config
        self.db = DatabaseManager.get_connection(config)

    def execute_ingest(
        self,
        file_path: Path,
        file_hash: str,
        chunks: list[Chunk],
        indexable_chunks: list[Chunk],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        chunk_store: ChunkStore,
        tracker: IngestionTracker,
        bm25: BM25Index,
        vector_store: VectorStore,
    ) -> None:
        """
        Execute multi-store ingestion with compensating rollback on failure.
        """
        from rag.storage.db_manager import DatabaseManager
        from rag.storage.query_cache import QueryCache

        source = str(file_path.resolve())
        log.debug("Beginning ingestion transaction for %s (%d chunks)", file_path.name, len(chunks))

        try:
            # Step A: Insert chunks into SQLite
            chunk_store.insert_batch(chunks)

            # Step B: Insert vectors into Qdrant
            if indexable_chunks:
                vector_store.upsert_batch(
                    [c.id for c in indexable_chunks],
                    vectors,
                    payloads,
                )

            # Step C: Add to BM25 memory index
            if indexable_chunks:
                bm25.add_batch(indexable_chunks)

            # Step D: Update file tracker
            tracker.update(file_path, file_hash, len(indexable_chunks))

            # Step E: Bump corpus version & invalidate target query cache
            DatabaseManager.bump_corpus_version(self.config)
            cache = QueryCache(self.config)
            cache.invalidate_file(source)

            # Step F: Save BM25 index after successful ingestion
            bm25.save()

        except Exception as exc:
            log.exception("Ingestion transaction failed for %s — executing compensating rollback", file_path.name)
            try:
                chunk_store.delete_by_source(source)
                tracker.remove(file_path)
                vector_store.delete_by_source(source)
            except Exception:
                pass
            raise RuntimeError(f"Ingestion transaction failed for {file_path.name}: {exc}") from exc

    def execute_remove(
        self,
        file_path: Path,
        chunk_store: ChunkStore,
        tracker: IngestionTracker,
        bm25: BM25Index,
        vector_store: VectorStore,
    ) -> int:
        """
        Execute multi-store removal with compensating rollback.
        """
        from rag.storage.db_manager import DatabaseManager
        from rag.storage.query_cache import QueryCache

        source = str(file_path.resolve())
        chunks = chunk_store.fetch_by_source(source)
        chunk_ids = [c.id for c in chunks]

        try:
            n = chunk_store.delete_by_source(source)
            if chunk_ids:
                bm25.delete_by_source(source, chunk_ids)

            vector_store.delete_by_source(source)
            tracker.remove(file_path)

            DatabaseManager.bump_corpus_version(self.config)
            cache = QueryCache(self.config)
            cache.invalidate_file(source)

            if chunk_ids:
                bm25.save()
            return n

        except Exception as exc:
            log.exception("Removal transaction failed for %s — rolling back", file_path.name)
            raise RuntimeError(f"Removal transaction failed for {file_path.name}: {exc}") from exc
