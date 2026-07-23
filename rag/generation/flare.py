"""
rag/generation/flare.py — Forward-Looking Active REtrieval (FLARE) Controller.

Implements iterative retrieval during generation (Phase 7-C).
Monitors token log probabilities; if confidence drops below a threshold,
pauses generation, retrieves fresh context using the most recently generated
sentence as a query, and resumes.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Generator

log = logging.getLogger(__name__)


class FlareController:
    """
    Wraps the LLM streaming process to inject context dynamically.
    """
    def __init__(
        self,
        llm: object,
        base_prompt: str,
        retrieve_fn: Callable[[str], str],
        confidence_threshold: float = -1.0,
        max_tokens: int = 150,
        temperature: float = 0.1,
    ) -> None:
        """
        Args:
            llm: LLMClient instance.
            base_prompt: Initial prompt with initial context.
            retrieve_fn: Callback that takes a search query and returns formatted text context.
            confidence_threshold: Logprob threshold (e.g., -1.0) below which to trigger retrieval.
        """
        self.llm = llm
        self.prompt = base_prompt
        self.retrieve_fn = retrieve_fn
        self.threshold = confidence_threshold
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.generated_text = ""
        self.tokens_yielded = 0
        self._sentence_buffer = ""
        
        # Regex to split sentences safely
        self._sentence_end = re.compile(r"([.!?])\s+")

    def stream(self) -> Generator[str, None, None]:
        """
        Stream tokens. Intercepts low-confidence tokens to run retrieval.
        """
        active_prompt = self.prompt
        
        while self.tokens_yielded < self.max_tokens:
            stream_gen = self.llm.stream(
                active_prompt, 
                max_tokens=self.max_tokens - self.tokens_yielded,
                temperature=self.temperature,
                return_logprobs=True
            )
            
            interrupted = False
            
            for item in stream_gen:
                if not isinstance(item, tuple):
                    # Should not happen since return_logprobs=True
                    continue
                    
                token, logprob = item
                
                # Check confidence (skip very short tokens like space or punctuation)
                if len(token.strip()) > 1 and logprob < self.threshold:
                    log.debug("FLARE: Low confidence token '%s' (%.2f). Pausing.", token, logprob)
                    interrupted = True
                    break
                    
                self.generated_text += token
                self._sentence_buffer += token
                self.tokens_yielded += 1
                yield token
                
                # If we naturally end a sentence with high confidence, clear buffer
                if self._sentence_end.search(self._sentence_buffer):
                    self._sentence_buffer = ""
                    
            if not interrupted:
                # Generation finished naturally
                break
                
            # --- Retrieval Triggered ---
            # Use the current sentence buffer (even if incomplete) + the low-confidence token as query
            search_query = self._sentence_buffer.strip()
            if not search_query:
                # If buffer is empty, just use the last few words generated
                words = self.generated_text.split()
                search_query = " ".join(words[-10:])
                
            log.info("FLARE triggering retrieval for query: %s", search_query)
            new_context = self.retrieve_fn(search_query)
            
            if not new_context.strip():
                log.debug("FLARE: No new context found.")
            else:
                log.debug("FLARE: Injecting new context.")
                # Append the new context to the prompt and instruct the model to continue
                active_prompt += (
                    f"\n\n[System: Additional context retrieved during generation]\n"
                    f"{new_context}\n"
                    f"[System: Continue the previous answer from exactly where you left off. "
                    f"Do not repeat what you have already said.]\n"
                    f"Continuing answer: {self.generated_text}"
                )
                
            self._sentence_buffer = ""
