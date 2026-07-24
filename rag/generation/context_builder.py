"""
rag/generation/context_builder.py — Assemble retrieved passages into LLM context.

Responsibilities:
  1. Token budget management — drop lowest-scoring passages first
  2. Anti-middle ordering — place highest-scoring passage first,
     second-highest last (mitigates Lost-in-the-Middle LLM attention bias)
  3. Prompt assembly — delegates to prompts.py

Reference: Liu et al. (2023) "Lost in the Middle: How Language Models Use Long Contexts"

Public API:
    ContextBuilder.build(passages, query, history, config) → (prompt, used_passages)
    ContextBuilder.build(passages, query, history, config) → (prompt, used_passages)
    _chronological_order(passages) → List[ScoredPassage]   (exported for testing)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag.types import Chunk, ScoredPassage

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

_WORDS_PER_TOKEN = 0.75


def _chronological_order(passages: list[ScoredPassage]) -> list[ScoredPassage]:
    """
    Group passages by source document and order them chronologically.
    
    Chronological ordering allows the LLM to follow step-by-step instructions
    and logical progression naturally, which is better for real-world tasks than
    anti-middle ordering.

    Args:
        passages: List of ScoredPassage (any order accepted).

    Returns:
        Reordered list sorted by source filename, then page/char offset/timestamp.
    """
    def sort_key(p: ScoredPassage):
        return (
            p.chunk.source,
            p.chunk.page or 0,
            p.chunk.char_start or 0,
            p.chunk.start_time or 0.0
        )
    return sorted(passages, key=sort_key)


def _merge_adjacent_chunks(passages: list[ScoredPassage]) -> list[ScoredPassage]:
    """
    Merge consecutive passages from the same source where page numbers are
    adjacent (N and N+1) or if they share the same file and sequence if pages are missing.
    Merged passage gets the higher score and method "merged".

    Only merges when both passages are from the same file and same source_type.
    Merged text = passage_A.text + "\\n\\n" + passage_B.text
    """
    if len(passages) <= 1:
        return passages

    merged = []
    i = 0
    while i < len(passages):
        current = passages[i]
        if i + 1 < len(passages):
            nxt = passages[i + 1]
            can_merge = False
            if current.chunk.source == nxt.chunk.source:
                if current.chunk.page is not None and nxt.chunk.page is not None:
                    if nxt.chunk.page == current.chunk.page + 1 or nxt.chunk.page == current.chunk.page:
                        can_merge = True
                else:
                    # If page info is missing but they are from the same source and close in char offsets
                    if abs(current.chunk.char_end - nxt.chunk.char_start) < 200 or abs(nxt.chunk.char_end - current.chunk.char_start) < 200:
                        can_merge = True

            if can_merge:
                merged_text = current.chunk.text + "\n\n" + nxt.chunk.text
                merged_chunk = Chunk(
                    id=current.chunk.id,  # keep first chunk's ID for citation
                    text=merged_text,
                    source=current.chunk.source,
                    filename=current.chunk.filename,
                    source_type=current.chunk.source_type,
                    page=current.chunk.page,
                    section=current.chunk.section or nxt.chunk.section,
                    char_start=min(current.chunk.char_start, nxt.chunk.char_start),
                    char_end=max(current.chunk.char_end, nxt.chunk.char_end),
                    token_count=current.chunk.token_count + nxt.chunk.token_count,
                    indexed_at=current.chunk.indexed_at,
                )
                merged.append(ScoredPassage(
                    chunk=merged_chunk,
                    score=max(current.score, nxt.score),
                    retrieval_method="merged",
                ))
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged


class ContextBuilder:
    """
    Assembles retrieved passages into an LLM-ready prompt.

    Thread-safe: no mutable state — each build() call is independent.
    """

    def build(
        self,
        passages: list[ScoredPassage],
        query: str,
        history: list[dict],
        config: RAGConfig,
    ) -> tuple[str, list[ScoredPassage]]:
        """
        Build the final LLM prompt from retrieved passages.

        Steps:
          1. Estimate token budget for passages
          2. Select passages that fit (drop lowest-scoring first)
          3. Apply anti-middle ordering to selected passages
          4. Assemble the prompt string

        Token budget estimation (word-count approximation, ratio 0.75 words/token):
          - Total tokens available:  config.generation.context_max_tokens
          - History consumes:        Σ words(turn) for each history turn
          - Prompt template overhead: ~200 tokens (RAG_PROMPT boilerplate)
          - Remaining:               for passage text

        The word approximation is intentionally conservative. It is better to
        exclude a marginal passage than to overflow the model's context window.

        Args:
            passages: Reranked candidates sorted by score descending.
            query:    The user's current question.
            history:  Rolling conversation history (may be empty).
            config:   RAGConfig for context_max_tokens.

        Returns:
            (prompt_str, passages_used)
            prompt_str:     Complete prompt to send to LLMClient.stream().
            passages_used:  Passages actually included (anti-middle ordered).
        """
        from rag.generation.prompts import build_prompt

        if not passages:
            return "", []

        import tiktoken
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            log.error("Failed to load tiktoken encoding, falling back to basic word count: %s", e)
            # Fallback for offline mode if tiktoken can't download
            tokenizer = None

        # ── Token budget ──────────────────────────────────────────────────────
        max_context = config.generation.context_max_tokens
        hard_limit = config.llm.ctx_size - config.llm.max_tokens - 50
        budget_tokens = min(max_context, hard_limit)

        if tokenizer:
            history_tokens = sum(len(tokenizer.encode(t.get("content", ""))) for t in history)
            overhead_tokens = len(tokenizer.encode(query)) + 200  # Prompt template overhead
        else:
            history_tokens = sum(len(t.get("content", "").split()) for t in history) * 1.3
            overhead_tokens = len(query.split()) * 1.3 + 200
            
        available_tokens = max(50, budget_tokens - history_tokens - overhead_tokens)

        selected: list[ScoredPassage] = []
        used_tokens = 0

        # passages is already sorted score-descending by the reranker
        for p in passages:
            if tokenizer:
                tokens = len(tokenizer.encode(p.chunk.text)) + 20 # Header overhead
            else:
                tokens = len(p.chunk.text.split()) * 1.3 + 20

            if used_tokens + tokens <= available_tokens:
                selected.append(p)
                used_tokens += tokens
            else:
                # Budget exhausted — stop adding
                break

        # Always include at least one passage (even if it overflows budget)
        if not selected:
            selected = [passages[0]]
            log.warning(
                "Context budget too small for any passage. "
                "Including passage 1 anyway (budget=%d tokens).",
                available_tokens,
            )

        log.debug(
            "ContextBuilder: %d/%d passages selected (%d tokens, budget %d)",
            len(selected),
            len(passages),
            used_tokens,
            available_tokens,
        )

        # ── Adjacent chunk merging ────────────────────────────────────────────
        # Sort by source and char_start before merging to ensure adjacent chunks are next to each other
        selected.sort(key=lambda p: (p.chunk.source, p.chunk.char_start))
        selected = _merge_adjacent_chunks(selected)
        
        # A1: Re-sort by score descending after merging
        selected.sort(key=lambda p: p.score, reverse=True)
        
        # ── Dynamic token budget enforcement (HIGH-02) ──────────────────────
        # Handled in the single exact token loop above using min(max_context, hard_limit).

        # ── Chronological ordering ────────────────────────────────────────────
        ordered = _chronological_order(selected)

        # ── Prompt assembly ───────────────────────────────────────────────
        prompt = build_prompt(query, ordered, history)

        return prompt, ordered
