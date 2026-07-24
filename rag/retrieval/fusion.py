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

from typing import TYPE_CHECKING

from rag.types import ScoredPassage

if TYPE_CHECKING:
    from rag.storage.chunk_store import ChunkStore

# Default RRF parameter. Higher k = less rank-sensitive fusion.
_DEFAULT_K = 60


def normalized_weighted_sum(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    top_k: int = 25,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists of (chunk_id, score) tuples using Normalized Weighted Summation
    (Convex Combination) instead of RRF.

    Args:
        ranked_lists: Each inner list is one ranking (e.g., dense, BM25).
        weights:      Weights for each list. Must sum to 1.0. Defaults to equal weighting.
        top_k:        Number of results to return.

    Returns:
        List of (chunk_id, fused_score) sorted descending by fused score.
        Length ≤ top_k. Returns [] if ranked_lists is empty.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
        
    if len(weights) != len(ranked_lists):
        raise ValueError("Length of weights must match length of ranked_lists")

    scores: dict[str, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        if not ranked_list:
            continue
            
        raw_scores = [score for _, score in ranked_list]
        max_score = max(raw_scores)
        min_score = min(raw_scores)
        score_range = max_score - min_score
        
        for chunk_id, score in ranked_list:
            if score_range > 0:
                norm_score = (score - min_score) / score_range
            else:
                norm_score = 1.0 if max_score > 0 else 0.0
                
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (norm_score * weight)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def fused_to_scored_passages(
    fused: list[tuple[str, float]],
    chunk_store: ChunkStore,
) -> list[ScoredPassage]:
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
