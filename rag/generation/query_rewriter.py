"""
rag/generation/query_rewriter.py — Fast local LLM query rewriting.

Translates conversational or imperative queries into dense keyword
phrases optimized for BM25 and cross-encoder scoring.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.generation.llm_client import LLMClient

log = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
Convert this into a concise keyword search phrase, max 8 words. Return only the phrase, nothing else.

Query: {query}
Phrase:"""

def rewrite_query(query: str, llm: LLMClient) -> str:
    """
    Rewrite a user query into a concise keyword search phrase.
    
    This is critical because cross-encoders (e.g. MS-MARCO) are trained on 
    Bing search queries and will catastrophically penalize imperative 
    commands like "list me the projects in...".
    
    Args:
        query: Raw user query
        llm:   Loaded LLMClient instance
        
    Returns:
        A concise keyword search string.
    """
    prompt = _REWRITE_PROMPT.format(query=query)
    
    try:
        rewritten = llm.generate(
            prompt,
            max_tokens=15,
            temperature=0.0,
        )
        
        # Clean up common artifacts from small LLMs (quotes, newlines)
        rewritten = rewritten.strip(" \"'\n")
        
        # Guard against the LLM failing completely and returning nothing
        if not rewritten:
            return query
            
        log.debug("Query rewritten: %r -> %r", query, rewritten)
        return rewritten
        
    except Exception as exc:
        log.warning("Query rewrite failed (%s) — using original query.", exc)
        return query
