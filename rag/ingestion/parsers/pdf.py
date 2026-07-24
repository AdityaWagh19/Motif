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

import io
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rag.ingestion.parsers.base import BaseParser, ParsedPage

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

# Heading detection: lines that look like section titles.
_MIN_HEADING_LEN = 5
_MAX_HEADING_LEN = 80
_MIN_HEADING_WORDS = 2
_MAX_HEADING_WORDS = 10
_TRAILING_PUNCT = (".", ",", ":", ";", "!", "?")

_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z]")


def _detect_section(text: str) -> str | None:
    """
    Heuristic section title detection for PDF pages.
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
    """

    SUPPORTED_EXTENSIONS = [".pdf"]

    def __init__(self, config: RAGConfig | None = None) -> None:
        self._config = config
        self._ocr = None

    def parse(self, path: Path) -> list[ParsedPage]:
        """
        Parse a PDF and return one ParsedPage per non-empty page.
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            import fitz  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "pymupdf is not installed. Run: pip install pymupdf"
            ) from exc

        pages: list[ParsedPage] = []

        try:
            doc = fitz.open(str(path))  # type: ignore[import]
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF {path}: {exc}") from exc

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            for page_num, page in enumerate(doc, start=1):  # type: ignore[call-overload]
                text: str = page.get_text("text").strip()  # type: ignore[union-attr]

                if not text:
                    if self._config and self._config.resolved_tier in ("T2", "T3"):
                        ocr_text = self._ocr_page(page, path)
                        if ocr_text:
                            pages.append(ParsedPage(
                                text=ocr_text,
                                page=page_num,
                                is_ocr=True,
                                has_image=True,
                            ))
                        continue
                    else:
                        log.debug(
                            "PDF page %d of %s has no text layer (scanned) — skipping (OCR requires T2+).",
                            page_num,
                            path.name,
                        )
                        continue

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
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:
                doc.close()  # type: ignore[union-attr]
            except Exception:
                pass

        if not pages:
            log.warning(
                "PDF %s produced no extractable text pages. "
                "The file may be scanned — OCR support comes in Phase 5.",
                path.name,
            )

        return pages

    def _ocr_page(self, fitz_page, doc_path: Path) -> str:
        """Export page as PNG and run PaddleOCR."""
        import os
        import tempfile

        import fitz  # type: ignore[import]
        
        mat = fitz.Matrix(2.0, 2.0)
        pix = fitz_page.get_pixmap(matrix=mat)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            pix.save(tmp_path)

        try:
            if self._ocr is None:
                from rag.ingestion.parsers.ocr import OCRPipeline
                self._ocr = OCRPipeline(self._config)
            return self._ocr.process_image(Path(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
