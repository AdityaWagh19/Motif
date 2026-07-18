"""
rag/retrieval/fusion.py — Reciprocal Rank Fusion (RRF) for multi-list ranking.

RRF combines ranked lists from heterogeneous retrieval methods (dense vector
search, BM25 lexical search) into a single merged ranking without requiring
score normalisation across systems.

Algorithm (Cormack et al., 2009):
    RRF(d) = Σᵢ 1 / (k + rank_i(d))

where rank_i(d) is the 1-based rank of chunk d in list i (absent → not counted).
k=60 is the standard value that prevents very-high-ranked items from dominating.

Public API:
    rrf_fuse(ranked_lists, top_k, k)         → List[(chunk_id, rrf_score)]
    rrf_to_scored_passages(fused, chunk_store) → List[ScoredPassage]
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from rag.types import ScoredPassage

if TYPE_CHECKING:
    from rag.storage.chunk_store import ChunkStore

# Default RRF parameter. Higher k = less rank-sensitive fusion.
_DEFAULT_K = 60


def rrf_fuse(
    ranked_lists: List[List[Tuple[str, float]]],
    top_k: int = 25,
    k: int = _DEFAULT_K,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists of (chunk_id, score) tuples using RRF.

    Args:
        ranked_lists: Each inner list is one ranking (dense, BM25, …).
                      Lists must be sorted descending by score.
                      Scores are not used in the fusion — only rank position.
        top_k:        Number of results to return.
        k:            RRF smoothing constant (default 60).

    Returns:
        List of (chunk_id, rrf_score) sorted descending by RRF score.
        Length ≤ top_k. Returns [] if ranked_lists is empty.
    """
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def rrf_to_scored_passages(
    fused: List[Tuple[str, float]],
    chunk_store: "ChunkStore",
) -> List[ScoredPassage]:
    """
    Fetch Chunk objects for fused IDs and wrap them in ScoredPassage.

    Chunks that no longer exist in ChunkStore (e.g., deleted mid-session)
    are silently dropped.

    Args:
        fused:       Output of rrf_fuse() — [(chunk_id, rrf_score), ...].
        chunk_store: ChunkStore instance to fetch chunk text + metadata from.

    Returns:
        List of ScoredPassage, same order as fused input, missing IDs dropped.
    """
    if not fused:
        return []

    chunk_ids = [cid for cid, _ in fused]
    score_map = dict(fused)
    chunks = chunk_store.fetch_batch(chunk_ids)

    return [
        ScoredPassage(
            chunk=c,
            score=score_map[c.id],
            retrieval_method="fused",
        )
        for c in chunks
        if c.id in score_map
    ]
