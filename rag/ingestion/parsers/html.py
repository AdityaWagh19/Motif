"""
rag/ingestion/parsers/html.py — BeautifulSoup-based HTML parser.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rag.ingestion.parsers.base import BaseParser, ParsedPage

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    """
    Parser for HTML files. Extracts main text, skipping scripts and styles.
    """

    SUPPORTED_EXTENSIONS = [".html", ".htm"]

    def __init__(self, config: RAGConfig | None = None) -> None:
        self._config = config

    def parse(self, path: Path) -> list[ParsedPage]:
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")

        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError(
                "BeautifulSoup is not installed. Run: uv pip install beautifulsoup4 lxml"
            ) from exc

        try:
            with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "lxml")
        except Exception as exc:
            raise RuntimeError(f"Failed to parse HTML {path}: {exc}") from exc

        # Remove scripts, styles, header, footer, nav
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        text = soup.get_text(separator="\n\n", strip=True)

        if not text:
            return []

        return [ParsedPage(text=text, has_table=bool(soup.find("table")))]
