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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig


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
    page: int | None = None          # 1-indexed; None for MD sections
    section: str | None = None       # Nearest heading above this content
    has_table: bool = False
    has_image: bool = False
    is_ocr: bool = False
    start_time: float | None = None  # Audio only (seconds)
    end_time: float | None = None    # Audio only (seconds)

    def __post_init__(self) -> None:
        import unicodedata
        # Normalise whitespace and unicode (ligatures, accents) at construction time
        self.text = unicodedata.normalize("NFKC", self.text.strip())


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
    SUPPORTED_EXTENSIONS: list[str] = []

    @abstractmethod
    def parse(self, path: Path) -> list[ParsedPage]:
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

#: File extensions supported across all parsers.
SUPPORTED_EXTENSIONS: list[str] = [
    ".pdf", ".md", ".txt", ".markdown", ".docx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
    ".html", ".htm", ".csv", ".tsv"
]


def get_parser(path: Path, config: RAGConfig | None = None) -> BaseParser:
    """
    Return the appropriate parser for the given file path.

    Parsers are tried in priority order: PDF → Markdown.

    Raises:
        ValueError: If no parser supports this file type.
    """
    # Import lazily so that missing third-party deps fail only at use time.
    from rag.ingestion.parsers.audio import AudioParser
    from rag.ingestion.parsers.docx import DOCXParser
    from rag.ingestion.parsers.image import ImageParser
    from rag.ingestion.parsers.markdown import MarkdownParser
    from rag.ingestion.parsers.pdf import PDFParser
    from rag.ingestion.parsers.html import HTMLParser
    from rag.ingestion.parsers.csv import CSVParser

    ext = path.suffix.lower()

    if ext == ".pdf":
        return PDFParser(config)
    if ext == ".docx":
        return DOCXParser()
    if ext in HTMLParser.SUPPORTED_EXTENSIONS:
        return HTMLParser(config)
    if ext in CSVParser.SUPPORTED_EXTENSIONS:
        return CSVParser(config)
    if ext in MarkdownParser.SUPPORTED_EXTENSIONS:
        return MarkdownParser()
    if ext in ImageParser.SUPPORTED_EXTENSIONS:
        if config is None:
            raise ValueError("config required for ImageParser")
        return ImageParser(config)
    if ext in AudioParser.SUPPORTED_EXTENSIONS:
        if config is None:
            raise ValueError("config required for AudioParser")
        return AudioParser(config)

    raise ValueError(
        f"No parser available for file type '{ext}'. "
        f"Supported: .pdf, .docx, .md, .txt, .png, .jpg, .mp3, .wav, .m4a, .html, .csv"
    )
