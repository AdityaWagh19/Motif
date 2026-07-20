"""
rag/retrieval/vector_store.py — Qdrant local-mode wrapper.

Manages the HNSW dense vector index. No Qdrant server is required — the
client operates entirely in-process using local files.

Phase 2: dense-only retrieval (768-dim cosine HNSW).
Phase 3: adds sparse vectors for full hybrid BM25 + dense + sparse search.

Collection: "motif_chunks"
Vector size: 768 (nomic-embed-text-v1.5)
Distance: Cosine
Storage: on_disk=True — vectors stored on disk, not RAM (T1 memory budget)
HNSW config: m=16, ef_construct=100 (quality / speed tradeoff)

Dependency: qdrant-client[local] >= 1.10

Dependency graph position:
    vector_store  →  qdrant_client  (third-party)
    vector_store  →  rag.config     (RAGConfig) — TYPE_CHECKING only
"""
from __future__ import annotations

import logging
import uuid as _uuid_module
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

COLLECTION_NAME_SUFFIX: str = "motif_chunks"
VECTOR_SIZE: int = 768


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Qdrant local-mode wrapper for dense vector search.

    Initialisation creates the Qdrant collection if it does not already exist.
    All vectors are stored on disk (on_disk=True) to respect the T1 ~5 GB
    memory footprint target.

    Usage:
        store = VectorStore(config)
        store.upsert_batch(chunk_ids, vectors, payloads)
        results = store.search_dense(query_vector, top_k=20)
    """

    def __init__(self, config: "RAGConfig") -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Run: pip install 'qdrant-client[local]'"
            ) from exc

        db_path = config.db_root / "qdrant"
        db_path.mkdir(parents=True, exist_ok=True)  # type: ignore[call-arg]
        self._collection_name = f"{config.storage.workspace}_{COLLECTION_NAME_SUFFIX}"
        self._client = QdrantClient(path=str(db_path))
        self._ensure_collection()
        log.debug("VectorStore initialised at %s, collection: %s", db_path, self._collection_name)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        existing_names = [
            c.name for c in self._client.get_collections().collections
        ]
        if self._collection_name not in existing_names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                hnsw_config={"m": 16, "ef_construct": 100, "on_disk": True},
            )
            log.info("Created Qdrant collection '%s'.", self._collection_name)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert(
        self,
        chunk_id: str,
        vector: np.ndarray,
        payload: Dict,
    ) -> None:
        """Insert or update a single vector with its metadata payload."""
        from qdrant_client.models import PointStruct  # type: ignore[import]

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=_str_to_uuid_int(chunk_id),
                    vector=vector.tolist(),
                    payload={**payload, "chunk_id": chunk_id},
                )
            ],
        )

    def upsert_batch(
        self,
        chunk_ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict],
    ) -> None:
        """
        Batch insert or update.

        Args:
            chunk_ids: List of chunk UUID strings.
            vectors:   (N, 768) float32 array.
            payloads:  List of N metadata dicts.
        """
        from qdrant_client.models import PointStruct  # type: ignore[import]

        if not chunk_ids:
            return

        BATCH_SIZE = 100
        for i in range(0, len(chunk_ids), BATCH_SIZE):
            batch_ids = chunk_ids[i:i + BATCH_SIZE]
            batch_vectors = vectors[i:i + BATCH_SIZE]
            batch_payloads = payloads[i:i + BATCH_SIZE]

            points = [
                PointStruct(
                    id=_str_to_uuid_int(cid),
                    vector=vec.tolist(),
                    payload={**payload, "chunk_id": cid},
                )
                for cid, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]
            self._client.upsert(collection_name=self._collection_name, points=points)

        log.debug("VectorStore: upserted %d points.", len(chunk_ids))

    # ------------------------------------------------------------------
    # Delete path
    # ------------------------------------------------------------------

    def delete_by_source(self, source: str) -> int:
        """
        Delete all vectors whose payload.source matches *source*.

        Args:
            source: Absolute file path string (as stored in chunk payload).

        Returns:
            0 — Qdrant does not return a count for filter-deletes.
            Callers should use ChunkStore.delete_by_source() for the count.
        """
        from qdrant_client.models import (  # type: ignore[import]
            Filter,
            FieldCondition,
            MatchValue,
            FilterSelector,
        )

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source),
                        )
                    ]
                )
            ),
        )
        log.debug("VectorStore: deleted points for source '%s'.", source)
        return 0

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filter_: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Dense HNSW approximate nearest-neighbour search.

        Args:
            query_vector: (768,) float32 query embedding.
            top_k:        Maximum number of results to return.
            filter_:      Optional metadata filter dict. Supported keys:
                            "source"      (str)  — exact match on file path
                            "source_type" (str)  — exact match on type string
                            "page_min"    (int)  — minimum page number (inclusive)
                            "page_max"    (int)  — maximum page number (inclusive)

        Returns:
            List of (chunk_id, cosine_score) tuples, sorted descending by score.
        """
        qdrant_filter = _build_filter(filter_) if filter_ else None
        
        limit = top_k if top_k is not None else 20

        if hasattr(self._client, "query_points"):
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector.tolist(),
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points
        else:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        return [(r.payload["chunk_id"], r.score) for r in results]  # type: ignore[index]

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        return self._client.count(collection_name=self._collection_name).count

    def close(self) -> None:
        """Close the underlying QdrantClient to release file locks."""
        if hasattr(self, "_client") and hasattr(self._client, "close"):
            self._client.close()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _str_to_uuid_int(s: str) -> int:
    """
    Convert a UUID string to an integer for use as a Qdrant point ID.

    Qdrant point IDs must be either unsigned 64-bit integers or UUID strings.
    We use the integer representation for consistency.
    """
    return _uuid_module.UUID(s).int


def _build_filter(filter_dict: Dict) -> Optional[object]:
    """
    Build a Qdrant Filter from a plain-dict specification.

    Returns None if no conditions are specified (avoids creating an empty filter).
    """
    from qdrant_client.models import (  # type: ignore[import]
        Filter,
        FieldCondition,
        MatchValue,
        Range,
    )

    conditions = []

    if "source" in filter_dict:
        conditions.append(
            FieldCondition(key="source", match=MatchValue(value=filter_dict["source"]))
        )
    if "source_type" in filter_dict:
        conditions.append(
            FieldCondition(
                key="source_type",
                match=MatchValue(value=filter_dict["source_type"]),
            )
        )
    if "page_min" in filter_dict or "page_max" in filter_dict:
        conditions.append(
            FieldCondition(
                key="page",
                range=Range(
                    gte=filter_dict.get("page_min"),
                    lte=filter_dict.get("page_max"),
                ),
            )
        )

    if not conditions:
        return None
    return Filter(must=conditions)
