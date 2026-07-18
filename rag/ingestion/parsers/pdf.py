"""
rag/ingestion/parsers/pdf.py — PyMuPDF-based PDF parser.

Extracts text page-by-page using fitz (PyMuPDF). Scanned pages (no text layer)
are skipped with a warning — OCR support is added in Phase 5.

Dependency: pymupdf (import as fitz)

Dependency graph position:
    pdf  →  fitz (pymupdf)
    pdf  →  rag.ingestion.parsers.base
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from rag.ingestion.parsers.base import BaseParser, ParsedPage

log = logging.getLogger(__name__)

# Heading detection: lines that look like section titles.
# Criteria: line ≤ 80 chars, no trailing sentence punctuation, 2–10 words.
_MIN_HEADING_LEN = 5
_MAX_HEADING_LEN = 80
_MIN_HEADING_WORDS = 2
_MAX_HEADING_WORDS = 10
_TRAILING_PUNCT = (".", ",", ":", ";", "!", "?")

# All-caps numbered section pattern: "1. INTRODUCTION" or "1.2 Methods"
_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z]")


def _detect_section(text: str) -> Optional[str]:
    """
    Heuristic section title detection for PDF pages.

    Scans lines from the top of the page text. Returns the first line that
    looks like a section heading:
      - Length between 5 and 80 characters
      - Does NOT end in sentence-terminating punctuation
      - Between 2 and 10 words

    Returns None if no heading-like line is found.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _MIN_HEADING_LEN <= len(line) <= _MAX_HEADING_LEN:
            if not line.endswith(_TRAILING_PUNCT):
                words = line.split()
                if _MIN_HEADING_WORDS <= len(words) <= _MAX_HEADING_WORDS:
                    return line
    return None


class PDFParser(BaseParser):
    """
    PyMuPDF-based parser for PDF files.

    Produces one ParsedPage per text-bearing PDF page. Scanned pages (empty
    text layer) are skipped — they will be processed by the OCR pipeline in
    Phase 5.
    """

    SUPPORTED_EXTENSIONS = [".pdf"]

    def parse(self, path: Path) -> List[ParsedPage]:
        """
        Parse a PDF and return one ParsedPage per non-empty page.

        Args:
            path: Path to the .pdf file.

        Returns:
            List of ParsedPage objects, one per page with extractable text.
            Empty (scanned) pages are omitted.

        Raises:
            FileNotFoundError: If path does not exist.
            RuntimeError:      If fitz cannot open or parse the file.
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            import fitz  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "pymupdf is not installed. Run: pip install pymupdf"
            ) from exc

        pages: List[ParsedPage] = []

        try:
            doc = fitz.open(str(path))  # type: ignore[import]
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF {path}: {exc}") from exc

        try:
            for page_num, page in enumerate(doc, start=1):  # type: ignore[call-overload]
                text: str = page.get_text("text").strip()  # type: ignore[union-attr]

                if not text:
                    # Scanned page — no text layer. Phase 5 will handle OCR.
                    log.debug(
                        "PDF page %d of %s has no text layer (scanned) — skipping.",
                        page_num,
                        path.name,
                    )
                    continue

                # Detect structural metadata
                try:
                    has_table = len(page.find_tables().tables) > 0  # type: ignore[union-attr]
                except Exception:
                    has_table = False

                try:
                    has_image = len(page.get_images()) > 0  # type: ignore[union-attr]
                except Exception:
                    has_image = False

                pages.append(
                    ParsedPage(
                        text=text,
                        page=page_num,
                        section=_detect_section(text),
                        has_table=has_table,
                        has_image=has_image,
                        is_ocr=False,
                    )
                )
        finally:
            doc.close()  # type: ignore[union-attr]

        if not pages:
            log.warning(
                "PDF %s produced no extractable text pages. "
                "The file may be scanned — OCR support comes in Phase 5.",
                path.name,
            )

        return pages
