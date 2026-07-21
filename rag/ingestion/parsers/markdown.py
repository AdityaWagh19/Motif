"""
rag/ingestion/parsers/markdown.py — Heading-aware Markdown/text parser.

Splits a Markdown document into sections delimited by H1/H2/H3 headings.
Each section becomes one ParsedPage whose `section` field is the heading text.
Plain text files (.txt) are treated as a single section.

Backend: markdown-it-py  (pip install markdown-it-py)

Dependency graph position:
    markdown  →  markdown_it (markdown-it-py)
    markdown  →  rag.ingestion.parsers.base
"""
from __future__ import annotations

import logging
from pathlib import Path

from rag.ingestion.parsers.base import BaseParser, ParsedPage

log = logging.getLogger(__name__)


def _is_heading_inline(tokens: list, current_token: object) -> bool:
    """
    Return True if *current_token* is the inline token carrying a heading's text.

    A heading inline token is immediately preceded by a heading_open token in
    the flat token list produced by markdown-it-py.
    """
    for i, tok in enumerate(tokens):
        if tok is current_token:
            return i > 0 and tokens[i - 1].type == "heading_open"
    return False


class MarkdownParser(BaseParser):
    """
    Heading-aware parser for Markdown (.md, .markdown) and plain text (.txt).

    Algorithm:
    1. Parse the file with markdown-it-py.
    2. Walk the flat token list.
    3. Each heading_open → inline sequence starts a new section.
    4. Inline tokens NOT immediately after a heading are accumulated as body text.
    5. On each new heading (or end of file), flush accumulated text into a ParsedPage.
    6. If no headings are found, the whole file is one ParsedPage.

    Returns:
        One ParsedPage per non-empty section.
    """

    SUPPORTED_EXTENSIONS = [".md", ".txt", ".markdown"]

    def parse(self, path: Path) -> list[ParsedPage]:
        """
        Parse a Markdown or plain-text file.

        Args:
            path: Path to the file.

        Returns:
            List of ParsedPage, one per heading-delimited section.
            If no headings exist, one page for the whole file.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            from markdown_it import MarkdownIt  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "markdown-it-py is not installed. Run: pip install markdown-it-py"
            ) from exc

        content = path.read_text(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        if not content.strip():
            log.debug("Markdown file %s is empty — returning no pages.", path.name)
            return []

        # Plain text files: no Markdown parsing needed — return as single page.
        if path.suffix.lower() == ".txt":
            text = content.strip()
            if text:
                return [ParsedPage(text=text, section=None)]
            return []

        md = MarkdownIt()
        tokens = md.parse(content)

        sections: list[ParsedPage] = []
        current_heading: str | None = None
        current_text_parts: list[str] = []

        def _flush() -> None:
            """Flush accumulated text as a new ParsedPage, if non-empty."""
            text = " ".join(current_text_parts).strip()
            if text:
                sections.append(ParsedPage(text=text, section=current_heading))

        for token in tokens:
            if token.type == "heading_open":
                # A new heading starts — flush whatever we accumulated so far.
                _flush()
                current_text_parts = []
                # heading text arrives in the next inline token (handled below)

            elif token.type == "inline" and token.content:
                if _is_heading_inline(tokens, token):
                    # This inline is the heading title itself.
                    current_heading = token.content.strip()
                else:
                    # Regular paragraph / list / code content.
                    current_text_parts.append(token.content)

        # Flush the final section.
        _flush()

        # Fallback: if markdown-it produced no structured sections, treat as raw text.
        if not sections and content.strip():
            sections = [ParsedPage(text=content.strip(), section=None)]

        # Filter out any empty-text pages (shouldn't happen after the guard above,
        # but defensive in case ParsedPage.__post_init__ strips to empty).
        result = [s for s in sections if s.text]
        log.debug(
            "MarkdownParser: %s → %d section(s)", path.name, len(result)
        )
        return result
