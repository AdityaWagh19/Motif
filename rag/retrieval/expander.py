"""
rag/retrieval/expander.py — Query expansion routing.

Phase 3: routing heuristic only. HyDE (Hypothetical Document Embeddings)
is implemented in Phase 4.

The QueryExpander wraps the embedding step so Phase 4 can transparently
swap in HyDE without changing any call sites in pipeline.py.

Public API:
    should_use_hyde(query, config) → bool
    QueryExpander.expand(query, config, embedder) → (vector, effective_query)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from rag.config import RAGConfig
    from rag.models.embedder import Embedder


def should_use_hyde(query: str, config: "RAGConfig") -> bool:
    """
    Decide whether to use HyDE for this query.

    Phase 3: always returns False.
    Phase 4 implements the heuristic:
      - config.retrieval.query_expansion == "hyde"
      - AND query length > 10 words (short factual queries benefit less)
      - AND tier is T2 or T3 (T1 latency budget too tight)

    Args:
        query:  The raw user query string.
        config: Loaded RAGConfig.

    Returns:
        False in Phase 3. True in Phase 4 when all conditions above are met.
    """
    return False


class QueryExpander:
    """
    Phase 3: thin wrapper — encodes the query directly.
    Phase 4: implements HyDE (generate a hypothetical answer, embed that).

    Using a class rather than a module-level function keeps the call site in
    pipeline.py identical between Phase 3 and Phase 4.
    """

    def expand(
        self,
        query: str,
        config: "RAGConfig",
        embedder: "Embedder",
    ) -> Tuple[np.ndarray, str]:
        """
        Expand a query for retrieval.

        Phase 3: direct embedding of the original query.
        Phase 4: if should_use_hyde(), generates a hypothetical answer with
                 LLMClient.generate() and embeds that instead.

        Args:
            query:    Raw user query string.
            config:   RAGConfig for tier / expansion settings.
            embedder: Loaded Embedder instance from ModelManager.

        Returns:
            (query_vector, effective_query_text)
            query_vector:       float32 numpy array of shape (embed_dim,).
            effective_query_text: The text that was actually embedded.
                                  In Phase 3 this is always the original query.
                                  In Phase 4 it may be the hypothetical answer.
        """
        # nomic-embed-text-v1.5 requires this prefix for query vectors
        vector = embedder.encode(query, prefix="search_query: ")
        return vector, query
