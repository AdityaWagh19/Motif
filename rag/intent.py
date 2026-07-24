"""
rag/intent.py — Non-restrictive Zero-Shot Intent Classification using Embeddings.

Combines a 0ms micro-heuristic guard with zero-shot vector similarity matching
using the pre-warmed nomic-embed-text-v1.5 embedder (~3ms latency).

Handles any phrasing variation (e.g. "hi, how are you?", "hello there good morning",
"what's up motif") without fragile regex rules or static set restrictions.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)


class Intent(Enum):
    QUERY = "query"
    CHITCHAT = "chitchat"
    GREETING_FAST = "greeting_fast"


# Exemplar anchor phrases for zero-shot vector similarity matching
_CHITCHAT_ANCHORS = [
    "hello hi hey good morning good evening how are you",
    "how are you doing what is up how is it going",
    "thank you thanks a lot great job good bot",
    "who are you what are you what can you do",
    "hi how are you doing today motif",
]

_QUERY_ANCHORS = [
    "what is the syllabus or topics covered in this document",
    "explain section 3 and detail the findings",
    "search the knowledge base for research results",
    "according to the report what are the requirements",
]


class IntentClassifier:
    """
    Zero-shot Intent Classifier combining ultra-fast heuristics with
    embedding vector similarity.
    """

    def __init__(self) -> None:
        self._chitchat_vectors: np.ndarray | None = None
        self._query_vectors: np.ndarray | None = None

        self._fast_greetings = {
            "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
            "how are you", "how are you doing", "what's up", "how's it going",
            "thanks", "thank you", "bye", "goodbye", "cheers"
        }

    def _init_anchor_vectors(self, embedder) -> None:
        """Lazily encode anchor centroids once on first embedder query."""
        if self._chitchat_vectors is not None and self._query_vectors is not None:
            return

        try:
            c_vecs = embedder.encode_batch(_CHITCHAT_ANCHORS, prefix="search_document: ")
            q_vecs = embedder.encode_batch(_QUERY_ANCHORS, prefix="search_document: ")

            # Normalize vectors for cosine similarity dot-product
            self._chitchat_vectors = c_vecs / (np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-9)
            self._query_vectors = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9)
            log.debug("IntentClassifier initialized zero-shot anchor vectors.")
        except Exception as exc:
            log.warning("Could not initialize intent classifier vectors: %s", exc)

    def classify(self, query: str, embedder=None) -> Intent:
        """
        Classify query intent in 0ms (heuristic) to ~3ms (vector similarity).

        Args:
            query: Raw query string from user.
            embedder: Optional loaded nomic embedder instance.
        """
        query_clean = re.sub(r"[^\w\s]", "", query.lower()).strip()
        tokens = query_clean.split()

        # ── 1. Micro-Heuristic Guard (0ms) ──────────────────────────────────
        if not tokens:
            return Intent.GREETING_FAST

        # Exact match for common greetings
        if query_clean in self._fast_greetings:
            log.info("Intent classified as GREETING_FAST (heuristic: %r)", query_clean)
            return Intent.GREETING_FAST

        # Check if query explicitly contains file extension / modifier flags
        if any(tok.startswith("/") for tok in query.split()):
            log.debug("Intent classified as QUERY (has inline modifier flag)")
            return Intent.QUERY

        # Direct short greetings like "hi there", "hello motif"
        if len(tokens) <= 3 and tokens[0] in ("hi", "hello", "hey", "greetings"):
            log.info("Intent classified as GREETING_FAST (short prefix match: %r)", query_clean)
            return Intent.GREETING_FAST

        # Conversational questions like "hi how are you", "how are you doing today"
        if ("how" in tokens or "what" in tokens) and any(w in tokens for w in ("you", "doing", "going", "up")):
            if not any(w in query_clean for w in ("syllabus", "document", "section", "page", "file", "code", "report", "pdf")):
                log.info("Intent classified as CHITCHAT (conversational phrasing: %r)", query_clean)
                return Intent.CHITCHAT

        # ── 2. Zero-Shot Embedding Vector Similarity (~3ms) ──────────────────
        if embedder is not None:
            try:
                self._init_anchor_vectors(embedder)
                if self._chitchat_vectors is not None and self._query_vectors is not None:
                    q_raw = embedder.encode_batch([query_clean], prefix="search_query: ")[0]
                    q_norm = q_raw / (np.linalg.norm(q_raw) + 1e-9)

                    # Compute max similarity to chitchat vs query anchors
                    c_sim = float(np.max(np.dot(self._chitchat_vectors, q_norm)))
                    q_sim = float(np.max(np.dot(self._query_vectors, q_norm)))

                    log.debug("Intent vector sim: chitchat=%.3f, query=%.3f", c_sim, q_sim)

                    # High similarity to chitchat anchor
                    if c_sim > 0.62 and c_sim >= q_sim:
                        log.info("Intent classified as CHITCHAT (zero-shot sim: %.3f)", c_sim)
                        return Intent.CHITCHAT
            except Exception as exc:
                log.debug("Zero-shot intent classification notice: %s", exc)

        # ── 3. Default Fallback ──────────────────────────────────────────────
        log.debug("Intent classified as QUERY (default)")
        return Intent.QUERY
