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
    _anti_middle_order(passages) → List[ScoredPassage]   (exported for testing)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

from rag.types import ScoredPassage

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)


def _anti_middle_order(passages: List[ScoredPassage]) -> List[ScoredPassage]:
    """
    Reorder passages so the most relevant content is at the extremes.

    LLMs attend more strongly to the beginning and end of their context window.
    This ordering ensures the two most relevant passages are not buried in the
    middle.

    Algorithm for N passages (sorted descending by score as [P1, P2, ..., PN]):
        Position 0   → P1  (highest score)
        Position N-1 → P2  (second highest score)
        Position 1   → P3  (third highest)
        Position 2   → P4  (fourth highest)
        ... etc.

    For N ≤ 2, the original order is returned unchanged (already optimal).

    Args:
        passages: List of ScoredPassage (any order accepted).

    Returns:
        Reordered list. Length equals input length.
    """
    if len(passages) <= 2:
        return passages[:]

    sorted_desc = sorted(passages, key=lambda p: p.score, reverse=True)
    result: List[ScoredPassage] = [None] * len(sorted_desc)  # type: ignore[list-item]

    # Best passage goes first, second-best goes last
    result[0] = sorted_desc[0]
    result[-1] = sorted_desc[1]

    # Fill middle positions with remaining passages (index 2 onward)
    middle_idx = 1
    for i in range(2, len(sorted_desc)):
        result[middle_idx] = sorted_desc[i]
        middle_idx += 1

    return result


class ContextBuilder:
    """
    Assembles retrieved passages into an LLM-ready prompt.

    Thread-safe: no mutable state — each build() call is independent.
    """

    def build(
        self,
        passages: List[ScoredPassage],
        query: str,
        history: List[dict],
        config: "RAGConfig",
    ) -> Tuple[str, List[ScoredPassage]]:
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

        # ── Token budget ──────────────────────────────────────────────────────
        # word count ≈ tokens × 0.75, so tokens ≈ words / 0.75
        # We work in words throughout to avoid any tokenizer dependency.
        budget_words = config.generation.context_max_tokens * 3 // 4
        history_words = sum(len(t["content"].split()) for t in history)
        overhead_words = 200  # RAG_PROMPT boilerplate + query
        available_words = budget_words - history_words - overhead_words

        selected: List[ScoredPassage] = []
        used_words = 0

        # passages is already sorted score-descending by the reranker
        for p in passages:
            words = len(p.chunk.text.split())
            if used_words + words <= available_words:
                selected.append(p)
                used_words += words
            else:
                # Budget exhausted — stop adding
                break

        # Always include at least one passage (even if it overflows budget)
        if not selected:
            selected = [passages[0]]
            log.warning(
                "Context budget too small for any passage. "
                "Including passage 1 anyway (budget=%d words).",
                available_words,
            )

        log.debug(
            "ContextBuilder: %d/%d passages selected (%d words, budget %d)",
            len(selected),
            len(passages),
            used_words,
            available_words,
        )

        # ── Anti-middle ordering ──────────────────────────────────────────────
        ordered = _anti_middle_order(selected)

        # ── Prompt assembly ───────────────────────────────────────────────────
        prompt = build_prompt(query, ordered, history)

        return prompt, ordered
