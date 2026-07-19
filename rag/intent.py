"""
rag/intent.py — Zero-shot Intent Classification using Embeddings.

Uses the existing Sentence-Transformer embedder to classify queries
as CHITCHAT vs QUERY based on cosine similarity to anchor phrases.
"""
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from rag.models.embedder import Embedder

log = logging.getLogger(__name__)

class Intent(Enum):
    QUERY = "query"
    CHITCHAT = "chitchat"
    GREETING_FAST = "greeting_fast"

class IntentClassifier:
    """
    Lightweight intent classifier.
    
    Computes cosine similarity between the user's query and a list of
    hardcoded chit-chat anchor phrases. If the similarity exceeds the
    threshold, it routes to CHITCHAT.
    """

    def __init__(self, embedder: "Embedder", threshold: float = 0.85):
        self._embedder = embedder
        self._threshold = threshold
        
        self._anchors = [
            "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
            "how are you", "how are you doing", "what's up", "how's it going",
            "thank you", "thanks", "thanks a lot", "thank you very much",
            "you are helpful", "you're so helpful", "great job", "good bot",
            "can you help me", "i need help", "i am confused"
        ]
        self._anchor_embeddings = None

    def _ensure_loaded(self):
        if self._anchor_embeddings is None:
            # We prefix anchors as 'search_query: ' to match the user query's semantic space.
            self._anchor_embeddings = self._embedder.encode_batch(
                self._anchors, prefix="search_query: "
            )
            log.debug("Pre-computed embeddings for %d chit-chat anchors", len(self._anchors))
            
    def classify(self, query: str) -> Intent:
        """
        Classify the query intent.
        """
        # Short-circuit: long queries are almost certainly document queries
        if len(query) > 120:
            return Intent.QUERY

        self._ensure_loaded()
        
        query_emb = self._embedder.encode(query, prefix="search_query: ")
        
        if self._anchor_embeddings is None:
            return Intent.QUERY
        
        # Dot product (since embeddings are L2 normalized, this is cosine similarity)
        similarities = np.dot(self._anchor_embeddings, query_emb)
        max_sim = float(np.max(similarities))
        
        query_lower = query.lower().strip()
        is_meta_question = query_lower in ["what are you", "what are you?", "who are you", "who are you?"]
        
        if max_sim > 0.92 and not is_meta_question:
            best_match = self._anchors[int(np.argmax(similarities))]
            log.info("Intent classified as GREETING_FAST (score: %.3f, matched: %r)", max_sim, best_match)
            return Intent.GREETING_FAST
        elif max_sim > self._threshold or is_meta_question:
            best_match = self._anchors[int(np.argmax(similarities))] if max_sim > self._threshold else "meta_question"
            log.info("Intent classified as CHITCHAT (score: %.3f, matched: %r)", max_sim, best_match)
            return Intent.CHITCHAT
            
        log.debug("Intent classified as QUERY (max chitchat score: %.3f)", max_sim)
        return Intent.QUERY
