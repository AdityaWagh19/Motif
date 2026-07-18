"""
tests/unit/test_parsers.py — Unit tests for PDF and Markdown parsers.

No model downloads or external services required.
PDF tests use unittest.mock to avoid needing a real PDF file for basic cases.
Markdown tests use the sample_md fixture from conftest.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from rag.ingestion.parsers.base import (
    BaseParser,
    ParsedPage,
    get_parser,
    SUPPORTED_EXTENSIONS,
)
from rag.ingestion.parsers.markdown import MarkdownParser
from rag.ingestion.parsers.pdf import PDFParser, _detect_section


# ---------------------------------------------------------------------------
# ParsedPage
# ---------------------------------------------------------------------------

class TestParsedPage:
    def test_text_is_stripped_on_init(self) -> None:
        p = ParsedPage(text="  hello world  ")
        assert p.text == "hello world"

    def test_defaults(self) -> None:
        p = ParsedPage(text="text")
        assert p.page is None
        assert p.section is None
        assert p.has_table is False
        assert p.has_image is False
        assert p.is_ocr is False


# ---------------------------------------------------------------------------
# _detect_section (PDF heuristic)
# ---------------------------------------------------------------------------

class TestDetectSection:
    def test_detects_short_heading(self) -> None:
        # 'Introduction Section' is 2 words, no trailing punctuation, ≤80 chars
        text = "Introduction Section\n\nThis is the body of the introduction section."
        result = _detect_section(text)
        assert result == "Introduction Section"

    def test_ignores_sentence_ending_with_period(self) -> None:
        text = "This is a sentence.\nAnother sentence here."
        # Lines ending with period should NOT be headings
        result = _detect_section(text)
        assert result is None

    def test_ignores_single_word_lines(self) -> None:
        # Single word ≡ < 2 words → not a heading
        result = _detect_section("Abstract\n\nBody text here with more words.")
        # "Abstract" is 1 word → not detected; "Body text here with more words" ends with . → not detected
        assert result is None

    def test_detects_numbered_heading(self) -> None:
        text = "1 Introduction\n\nThis section covers the basics."
        result = _detect_section(text)
        assert result == "1 Introduction"

    def test_none_for_empty_text(self) -> None:
        assert _detect_section("") is None

    def test_ignores_very_long_lines(self) -> None:
        long_line = "A " * 50  # 100 chars → > 80 char limit
        result = _detect_section(long_line)
        assert result is None


# ---------------------------------------------------------------------------
# PDFParser
# ---------------------------------------------------------------------------

class TestPDFParser:
    def test_supported_extensions(self) -> None:
        assert ".pdf" in PDFParser.SUPPORTED_EXTENSIONS

    def test_can_parse_pdf(self) -> None:
        assert PDFParser.can_parse(Path("document.pdf"))

    def test_cannot_parse_md(self) -> None:
        assert not PDFParser.can_parse(Path("notes.md"))

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nonexistent.pdf")

    def test_skips_empty_pages(self, tmp_path: Path) -> None:
        """Pages with no text layer should be omitted from output."""
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")  # placeholder — fitz mock handles it

        # Mock fitz so we control what pages return
        mock_page_empty = MagicMock()
        mock_page_empty.get_text.return_value = ""        # scanned page
        mock_page_empty.find_tables.return_value.tables = []
        mock_page_empty.get_images.return_value = []

        mock_page_text = MagicMock()
        mock_page_text.get_text.return_value = "  Real text content here.  "
        mock_page_text.find_tables.return_value.tables = []
        mock_page_text.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page_empty, mock_page_text]))
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            pages = PDFParser().parse(pdf_path)

        assert len(pages) == 1
        assert pages[0].text == "Real text content here."
        assert pages[0].page == 2  # second page in the document

    def test_detects_tables_and_images(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rich.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Table data here"
        mock_page.find_tables.return_value.tables = [MagicMock()]   # 1 table
        mock_page.get_images.return_value = [MagicMock(), MagicMock()]  # 2 images

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            pages = PDFParser().parse(pdf_path)

        assert len(pages) == 1
        assert pages[0].has_table is True
        assert pages[0].has_image is True

    def test_returns_empty_list_for_all_scanned(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "all_scanned.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.get_text.return_value = ""

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            pages = PDFParser().parse(pdf_path)

        assert pages == []


# ---------------------------------------------------------------------------
# MarkdownParser
# ---------------------------------------------------------------------------

class TestMarkdownParser:
    def test_supported_extensions(self) -> None:
        for ext in [".md", ".txt", ".markdown"]:
            assert ext in MarkdownParser.SUPPORTED_EXTENSIONS

    def test_can_parse_md(self) -> None:
        assert MarkdownParser.can_parse(Path("notes.md"))

    def test_cannot_parse_pdf(self) -> None:
        assert not MarkdownParser.can_parse(Path("doc.pdf"))

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MarkdownParser().parse(tmp_path / "missing.md")

    def test_extracts_sections(self, sample_md: Path) -> None:
        pages = MarkdownParser().parse(sample_md)
        assert len(pages) >= 2
        headings = [p.section for p in pages if p.section]
        assert len(headings) >= 1

    def test_all_pages_have_non_empty_text(self, sample_md: Path) -> None:
        pages = MarkdownParser().parse(sample_md)
        for p in pages:
            assert p.text.strip(), f"Empty text in page with section={p.section!r}"

    def test_section_matches_first_heading(self, sample_md: Path) -> None:
        pages = MarkdownParser().parse(sample_md)
        # sample_md starts with "# Introduction"
        headings = [p.section for p in pages if p.section]
        assert any("Introduction" in h for h in headings)

    def test_no_heading_file_is_one_page(self, tmp_path: Path) -> None:
        md = tmp_path / "flat.md"
        md.write_text("Just some text here.\nAnd more text.", encoding="utf-8")
        pages = MarkdownParser().parse(md)
        assert len(pages) == 1
        assert pages[0].section is None

    def test_empty_file_returns_no_pages(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        pages = MarkdownParser().parse(md)
        assert pages == []

    def test_txt_file_is_single_page(self, tmp_path: Path) -> None:
        txt = tmp_path / "note.txt"
        txt.write_text("Hello world. This is plain text.", encoding="utf-8")
        pages = MarkdownParser().parse(txt)
        assert len(pages) == 1
        assert pages[0].section is None

    def test_markdown_with_multiple_headings(self, tmp_path: Path) -> None:
        md = tmp_path / "multi.md"
        md.write_text(
            "# Chapter One\n\nFirst chapter content.\n\n"
            "## Section 1.1\n\nSection content here.\n\n"
            "# Chapter Two\n\nSecond chapter content.\n",
            encoding="utf-8",
        )
        pages = MarkdownParser().parse(md)
        assert len(pages) >= 2
        sections = [p.section for p in pages if p.section]
        assert any("Chapter One" in s for s in sections)


# ---------------------------------------------------------------------------
# get_parser registry
# ---------------------------------------------------------------------------

class TestGetParser:
    def test_get_parser_pdf(self) -> None:
        parser = get_parser(Path("document.pdf"))
        assert isinstance(parser, PDFParser)

    def test_get_parser_md(self) -> None:
        parser = get_parser(Path("notes.md"))
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_txt(self) -> None:
        parser = get_parser(Path("readme.txt"))
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_markdown_extension(self) -> None:
        parser = get_parser(Path("doc.markdown"))
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="No parser available"):
            get_parser(Path("video.mp4"))

    def test_get_parser_case_insensitive(self) -> None:
        parser = get_parser(Path("DOC.PDF"))
        assert isinstance(parser, PDFParser)
