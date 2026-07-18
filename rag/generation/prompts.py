"""
rag/generation/prompts.py — LLM prompt templates and formatting utilities.

All string manipulation for the generation step lives here. Nothing in this
module does I/O, loads models, or calls external services — pure functions only.

Constants:
    RAG_PROMPT              — main RAG prompt template
    HISTORY_SYSTEM_PROMPT   — used when conversation history is present
    HYDE_PROMPT             — Phase 4: generate a hypothetical answer for HyDE

Functions:
    format_context(passages)              → numbered context string
    format_history(history)              → User:/Assistant: block string
    build_prompt(query, passages, history) → assembled LLM prompt string
    build_citations(passages)             → List[Citation]
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from rag.types import Citation

if TYPE_CHECKING:
    from rag.types import ScoredPassage


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

# System prompt injected at the start of every query.
# {context} → assembled retrieved passages (numbered).
# {query}   → the user's question.
RAG_PROMPT = """\
You are a precise research assistant. Answer the question using ONLY the \
information in the provided context passages. Do not speculate or use outside \
knowledge.

If the answer is not present in the context, say exactly:
"I cannot find an answer to this in the available documents."

Cite each piece of information with its passage number in square brackets, \
e.g. [1], [2]. If multiple passages support a point, cite all of them, e.g. [1][3].

Context:
{context}

Question: {query}
Answer:"""


# System prompt used when conversation history is present.
# {history} → prior turns formatted as User:/Assistant: blocks.
HISTORY_SYSTEM_PROMPT = """\
You are a precise research assistant continuing a conversation. Prior context:

{history}

Answer the current question using ONLY the provided document passages. \
Maintain consistency with your previous answers. Cite sources with [N]."""


# Prompt used by HyDE (Phase 4) to generate a hypothetical answer.
HYDE_PROMPT = """\
Write a short, factual passage (2-3 sentences) that would answer the following \
question if it existed in a research document. Focus on the most likely answer.

Question: {query}
Passage:"""


# ─────────────────────────────────────────────────────────────────────────────
# Formatting utilities
# ─────────────────────────────────────────────────────────────────────────────

def format_context(passages: "List[ScoredPassage]") -> str:
    """
    Format a list of ScoredPassage objects into a numbered context string.

    Output format:
        [1] Source: filename.pdf (p.3, Section Title)
        <passage text>

        [2] Source: notes.md (Introduction)
        <passage text>
        ...

    Args:
        passages: List of passages to include in context (ordered for LLM).

    Returns:
        Multi-line string ready to be inserted into RAG_PROMPT {context}.
        Returns empty string if passages is empty.
    """
    if not passages:
        return ""

    lines: List[str] = []
    for i, p in enumerate(passages, start=1):
        chunk = p.chunk
        loc_parts: List[str] = []
        if chunk.page is not None:
            loc_parts.append(f"p.{chunk.page}")
        if chunk.section:
            loc_parts.append(chunk.section)
        loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
        lines.append(f"[{i}] Source: {chunk.filename}{loc}")
        lines.append(chunk.text.strip())
        lines.append("")

    return "\n".join(lines).strip()


def format_history(history: List[dict]) -> str:
    """
    Format conversation history as alternating User/Assistant blocks.

    Args:
        history: List of {"role": "user"|"assistant", "content": str} dicts,
                 oldest-first.

    Returns:
        Multi-line string with User: / Assistant: prefixes.
        Returns empty string if history is empty.
    """
    if not history:
        return ""

    parts: List[str] = []
    for turn in history:
        prefix = "User" if turn["role"] == "user" else "Assistant"
        parts.append(f"{prefix}: {turn['content']}")
    return "\n\n".join(parts)


def build_prompt(
    query: str,
    passages: "List[ScoredPassage]",
    history: List[dict],
) -> str:
    """
    Assemble the final LLM prompt from context, query, and history.

    If history is non-empty: prepend HISTORY_SYSTEM_PROMPT + context block.
    If no history: use RAG_PROMPT only.

    Args:
        query:    The user's current question.
        passages: Passages in display order (already anti-middle reordered).
        history:  Rolling history window (may be empty list).

    Returns:
        Complete prompt string ready to be sent to LLMClient.stream().
    """
    context = format_context(passages)

    if history:
        history_text = format_history(history)
        system = HISTORY_SYSTEM_PROMPT.format(history=history_text)
        return (
            f"{system}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
    else:
        return RAG_PROMPT.format(context=context, query=query)


def build_citations(passages: "List[ScoredPassage]") -> List[Citation]:
    """
    Build Citation objects from the final reranked passages.

    Citation numbers correspond to [N] markers in the generated answer.

    Args:
        passages: Passages in display order (same order as format_context).

    Returns:
        List of Citation, one per passage, numbered 1-based.
    """
    citations: List[Citation] = []
    for i, p in enumerate(passages, start=1):
        c = p.chunk
        citations.append(
            Citation(
                number=i,
                source_type=c.source_type,
                filepath=c.source,
                filename=c.filename,
                page=c.page,
                section=c.section,
                start_time=c.start_time,
                end_time=c.end_time,
                relevance_score=p.score,
                excerpt=c.text[:150],
            )
        )
    return citations
