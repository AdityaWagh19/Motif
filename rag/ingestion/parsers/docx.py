"""
rag/ingestion/parsers/docx.py — Parser for DOCX files using python-docx.

Extracts text, headings, and tables from .docx files.
Converts tables into markdown format. Uses headings to split sections.
"""
from __future__ import annotations

import logging
from pathlib import Path

try:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError:
    docx = None

from rag.ingestion.parsers.base import BaseParser, ParsedPage

log = logging.getLogger(__name__)

class DOCXParser(BaseParser):
    """
    Parser for Microsoft Word (.docx) files.
    Requires python-docx.
    """
    
    SUPPORTED_EXTENSIONS = [".docx"]

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    def parse(self, path: Path) -> list[ParsedPage]:
        if docx is None:
            raise ImportError(
                "python-docx is required to parse .docx files. "
                "Install it with `pip install python-docx`"
            )

        try:
            doc = docx.Document(str(path))
        except Exception as e:
            log.error("Failed to parse DOCX %s: %s", path, e)
            return []

        sections: list[ParsedPage] = []
        current_heading: str | None = None
        current_parts: list[str] = []
        current_has_table = False

        def flush():
            nonlocal current_parts, current_has_table
            if current_parts:
                text = "\n".join(current_parts).strip()
                if text:
                    sections.append(ParsedPage(
                        text=text,
                        section=current_heading,
                        has_table=current_has_table,
                        page=1, # DOCX parsing doesn't easily yield page numbers
                    ))
            current_parts = []
            current_has_table = False

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                para = Paragraph(element, doc)
                style_name = (para.style.name or "") if para.style else ""
                text = para.text.strip()

                if style_name.startswith("Heading"):
                    flush()
                    current_heading = text or current_heading
                elif text:
                    current_parts.append(text)

            elif tag == "tbl":
                table = Table(element, doc)
                md_table = self._table_to_markdown(table)
                if md_table:
                    current_parts.append(md_table)
                    current_has_table = True

        flush()
        return sections

    def _table_to_markdown(self, table) -> str:
        """Convert a python-docx Table to a markdown pipe table string."""
        rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(rows) if rows else ""
