"""
rag/intent.py — Zero-shot Intent Classification using Embeddings.

Uses the existing Sentence-Transformer embedder to classify queries
as CHITCHAT vs QUERY based on cosine similarity to anchor phrases.
"""
import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

class Intent(Enum):
    QUERY = "query"
    CHITCHAT = "chitchat"
    GREETING_FAST = "greeting_fast"

class IntentClassifier:
    """
    Lightweight intent classifier using fast heuristics.
    
    Checks for obvious conversational inputs (greetings, short phrases) 
    using keyword matching and simple rules to avoid the latency of embeddings.
    """

    def __init__(self):
        # We no longer need an embedder for fast heuristics
        
        self._fast_greetings = {
            "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
            "how are you", "how are you doing", "what's up", "how's it going"
        }
        
        self._conversational_phrases = {
            "thank you", "thanks", "thanks a lot", "thank you very much",
            "you are helpful", "you're so helpful", "great job", "good bot",
            "can you help me", "i need help", "i am confused",
            "let's do some work", "are you ready", "hello, should we do some work",
            "let's get started", "what can you do", "who are you", "what are you",
            "hey lets get some work done"
        }
        
    def classify(self, query: str) -> Intent:
        """
        Classify the query intent using fast heuristics.
        """
        query_lower = query.lower().strip()
        
        # Remove punctuation for matching
        import re
        query_clean = re.sub(r'[^\w\s]', '', query_lower)
        
        # 1. Exact match for fast greetings
        if query_clean in self._fast_greetings:
            log.info("Intent classified as GREETING_FAST (matched: %r)", query_clean)
            return Intent.GREETING_FAST
            
        # 2. Exact match for conversational phrases
        if query_clean in self._conversational_phrases:
            log.info("Intent classified as CHITCHAT (matched: %r)", query_clean)
            return Intent.CHITCHAT
            
        # 3. Meta questions
        is_meta_question = query_lower in ["what are you", "what are you?", "who are you", "who are you?"]
        if is_meta_question:
            log.info("Intent classified as CHITCHAT (meta question)")
            return Intent.CHITCHAT
            
        # 4. Short conversational checks
        word_count = len(query_clean.split())
        if word_count < 5:
            # If it's very short and contains conversational words but no clear nouns,
            # we might route to chitchat. But for safety, default to QUERY 
            # and let retrieval confidence handle the rest.
            pass
            
        log.debug("Intent classified as QUERY (heuristic fallback)")
        return Intent.QUERY
