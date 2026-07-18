"""
rag/retrieval/expander.py — Query expansion with HyDE routing.

Phase 4: HyDE (Hypothetical Document Embeddings) fully implemented.

should_use_hyde() heuristic:
  - config.retrieval.query_expansion == "hyde"
  - AND query has > 10 words (short factual queries benefit less from HyDE)
  - AND tier is T2 or T3 (T1 latency budget too tight for extra LLM call)

QueryExpander.expand():
  - Phase 3: direct query embedding
  - Phase 4: if should_use_hyde() → generate hypothetical answer → embed it

Public API:
    should_use_hyde(query, config) → bool
    QueryExpander.expand(query, config, embedder) → (vector, effective_query)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from rag.config import RAGConfig
    from rag.models.embedder import Embedder

log = logging.getLogger(__name__)

# HyDE prompt: ask the LLM to write a hypothetical passage that would
# answer the query. We then embed the passage instead of the query.
_HYDE_PROMPT = """\
Write a factual passage of 2-3 sentences that would directly answer the \
following question if it appeared in a research document. Be specific and \
concise. Do NOT include the question itself.

Question: {query}
Passage:"""

# Minimum word count for HyDE to be triggered.
_HYDE_MIN_WORDS = 8


def should_use_hyde(query: str, config: "RAGConfig") -> bool:
    """
    Decide whether to use HyDE for this query.

    Conditions (all must be true):
      1. config.retrieval.query_expansion == "hyde"
      2. Query has > _HYDE_MIN_WORDS words (short queries benefit less)
      3. Resolved tier is T2 or T3 (T1 latency too tight for extra LLM call)

    Args:
        query:  The raw user query string.
        config: Loaded RAGConfig.

    Returns:
        True if HyDE should be used, False otherwise.
    """
    if getattr(config.retrieval, "query_expansion", "none") != "hyde":
        return False
    if len(query.split()) <= _HYDE_MIN_WORDS:
        return False
    tier = getattr(config, "resolved_tier", "T1")
    if tier == "T1":
        return False
        
    FACTUAL_PREFIXES = (
        "who is", "who was", "what is", "what are", "what was",
        "when did", "when was", "where is", "where was", "how many",
        "how much", "list ", "name ", "define ",
    )
    query_lower = query.lower()
    if any(query_lower.startswith(p) for p in FACTUAL_PREFIXES):
        return False
        
    return True


class QueryExpander:
    """
    Phase 4: implements HyDE query expansion transparently.

    The class design means pipeline.py call sites are unchanged between
    Phase 3 (direct embed) and Phase 4 (HyDE embed).
    """

    def expand(
        self,
        query: str,
        config: "RAGConfig",
        embedder: "Embedder",
    ) -> Tuple[np.ndarray, str]:
        """
        Expand a query for retrieval.

        Phase 3 path: direct embedding of the original query.
        Phase 4 path: if should_use_hyde(), generate a hypothetical answer
                      with the LLM and embed that instead.

        Args:
            query:    Raw user query string.
            config:   RAGConfig for tier / expansion settings.
            embedder: Loaded Embedder instance from ModelManager.

        Returns:
            (query_vector, effective_query_text)
            query_vector:         float32 numpy array of shape (embed_dim,).
            effective_query_text: The text that was actually embedded.
                                  Original query (Phase 3) or hypothetical
                                  answer (Phase 4 HyDE path).
        """
        if should_use_hyde(query, config):
            try:
                effective_query = self._generate_hypothetical(query, config)
                log.debug(
                    "HyDE: embedded hypothetical answer (%.80s…)", effective_query
                )
            except Exception as exc:
                # If HyDE fails (LLM not loaded, OOM, etc.) fall back gracefully.
                log.warning(
                    "HyDE generation failed (%s) — falling back to direct query.", exc
                )
                effective_query = query
        else:
            effective_query = query

        # nomic-embed-text-v1.5 requires this prefix for query vectors.
        vector = embedder.encode(effective_query, prefix="search_query: ")
        return vector, effective_query

    def _generate_hypothetical(self, query: str, config: "RAGConfig") -> str:
        """
        Generate a hypothetical document passage using the loaded LLM.

        Args:
            query:  The user's query.
            config: RAGConfig (used to fetch the LLM from ModelManager).

        Returns:
            A 2-3 sentence hypothetical passage string.

        Raises:
            FileNotFoundError: If the LLM model file is not downloaded.
            RuntimeError:      If the LLM is not loaded.
        """
        from rag.models.model_manager import get_model_manager

        llm = get_model_manager().get_llm(config)
        prompt = _HYDE_PROMPT.format(query=query)

        # 150 tokens ≈ 2-3 sentences — enough for a good hypothetical.
        return llm.generate(
            prompt,
            max_tokens=150,
            temperature=0.3,  # slightly higher than answer temp — more creative
        ).strip()
