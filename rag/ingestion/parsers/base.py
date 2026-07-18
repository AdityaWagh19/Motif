"""
rag/ingestion/parsers/base.py — BaseParser ABC and parser registry.

All document parsers implement BaseParser. Callers use get_parser(path)
to obtain the right parser for a given file — never instantiate parsers directly.

Dependency graph position:
    base  →  (stdlib only)
    pdf   →  pymupdf (fitz)
    markdown → markdown-it-py
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedPage:
    """
    One logical unit from a parsed document.

    For PDFs: one page.
    For Markdown/text: one section (delimited by headings).
    For audio: one transcript segment (Phase 5).

    text must always be non-empty (parsers must not return empty-text pages).
    """
    text: str
    page: Optional[int] = None          # 1-indexed; None for MD sections
    section: Optional[str] = None       # Nearest heading above this content
    has_table: bool = False
    has_image: bool = False
    is_ocr: bool = False
    start_time: Optional[float] = None  # Audio only (seconds)
    end_time: Optional[float] = None    # Audio only (seconds)

    def __post_init__(self) -> None:
        # Normalise whitespace at construction time so every caller gets clean text
        self.text = self.text.strip()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseParser(ABC):
    """
    All parsers produce a list of ParsedPage from a file path.

    Parsers do NOT chunk — they return page/section-level units.
    Chunking is handled by chunker.py.
    """

    #: File extensions this parser handles, e.g. [".pdf"]
    SUPPORTED_EXTENSIONS: List[str] = []

    @abstractmethod
    def parse(self, path: Path) -> List[ParsedPage]:
        """
        Parse a file and return one ParsedPage per logical document unit.

        Args:
            path: Absolute path to the file.

        Returns:
            Non-empty list of ParsedPage objects.
            Each ParsedPage.text must be non-empty after stripping.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError:        If the file type is not supported by this parser.
            RuntimeError:      If parsing fails for any other reason.
        """
        ...

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        """Return True if this parser handles the given file extension."""
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

#: File extensions supported across all Phase 2 parsers.
SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".md", ".txt", ".markdown"]


def get_parser(path: Path) -> BaseParser:
    """
    Return the appropriate parser for the given file path.

    Parsers are tried in priority order: PDF → Markdown.

    Raises:
        ValueError: If no parser supports this file type.
    """
    # Import lazily so that missing third-party deps fail only at use time.
    from rag.ingestion.parsers.pdf import PDFParser
    from rag.ingestion.parsers.markdown import MarkdownParser

    for parser_class in [PDFParser, MarkdownParser]:
        if parser_class.can_parse(path):
            return parser_class()

    raise ValueError(
        f"No parser available for file type '{path.suffix}'. "
        f"Supported in Phase 2: .pdf, .md, .txt, .markdown"
    )
