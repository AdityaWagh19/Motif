"""
rag/reranking/cross_encoder.py — Passage reranking using the cross-encoder.

Does NOT own model loading — uses ModelManager.get_reranker().

Algorithm:
  1. Score all candidate (query, passage) pairs via Reranker.score()
  2. Filter: drop passages with score < threshold
  3. Sort descending by score
  4. Return top_k, with ScoredPassage.score replaced by cross-encoder score

Public API:
    rerank(query, passages, config, top_k, threshold) → List[ScoredPassage]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag.types import ScoredPassage

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)


def rerank(
    query: str,
    passages: list[ScoredPassage],
    config: RAGConfig,
    top_k: int = 5,
    threshold: float = 0.3,
) -> list[ScoredPassage]:
    """
    Rerank a list of ScoredPassage objects using the cross-encoder.

    Args:
        query:     The user's query string.
        passages:  Pre-retrieved candidates from RRF fusion.
        config:    RAGConfig — used to obtain the ModelManager and thresholds.
        top_k:     Maximum number of passages to return.
        threshold: Minimum relevance score. Passages below this are dropped.
                   Use config.retrieval.relevance_threshold in the pipeline.

    Returns:
        Reranked list (length ≤ top_k) sorted by cross-encoder score descending.
        May be empty if all passages score below the threshold.
    """
    if not passages:
        return []

    from rag.models.model_manager import get_model_manager

    reranker = get_model_manager().get_reranker(config)
    texts = [p.chunk.text for p in passages]

    log.debug("Reranking %d candidates for query: %.60s…", len(passages), query)
    scores = reranker.score(query, texts)
    
    # Adaptive thresholding
    ABSOLUTE_FLOOR = 0.01
    ADAPTIVE_RATIO = 0.5
    
    max_score = float(scores.max())
    adaptive_threshold = max(ABSOLUTE_FLOOR, min(threshold, max_score * ADAPTIVE_RATIO))

    reranked: list[ScoredPassage] = []
    for passage, score in zip(passages, scores):
        float_score = float(score)
        if float_score >= adaptive_threshold:
            reranked.append(
                ScoredPassage(
                    chunk=passage.chunk,
                    score=float_score,
                    retrieval_method="reranked",
                )
            )

    reranked.sort(key=lambda p: p.score, reverse=True)
    result = reranked[:top_k]

    log.debug(
        "Reranker adaptive_threshold: %.3f (max_score: %.3f, static_threshold: %.3f)",
        adaptive_threshold, max_score, threshold
    )
    log.debug(
        "Reranker: %d/%d passages above adaptive threshold",
        len(result),
        len(passages)
    )
    return result
