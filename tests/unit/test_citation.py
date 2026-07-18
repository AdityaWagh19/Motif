"""
tests/unit/test_citation.py — Unit tests for citation formatting.
"""
from __future__ import annotations

from rag.types import Citation

def test_citation_format_pdf() -> None:
    c = Citation(
        number=1,
        source_type="pdf",
        filepath="/docs/thesis.pdf",
        filename="thesis.pdf",
        page=42,
        section="Methods"
    )
    assert c.format() == "[1] thesis.pdf (p.42) — Methods"

def test_citation_format_md() -> None:
    c = Citation(
        number=2,
        source_type="md",
        filepath="/notes/ideas.md",
        filename="ideas.md",
        section="Introduction"
    )
    # Markdown citations don't have pages, only sections if available.
    assert c.format() == "[2] ideas.md"

def test_citation_format_audio() -> None:
    c = Citation(
        number=3,
        source_type="audio",
        filepath="/rec/talk.mp3",
        filename="talk.mp3",
        start_time=125.0,
        end_time=137.5
    )
    # 125s = 02:05, 137.5s = 02:17
    assert c.format() == "[3] talk.mp3 @ 02:05–02:17"

def test_citation_format_audio_no_end_time() -> None:
    c = Citation(
        number=4,
        source_type="audio",
        filepath="/rec/talk.mp3",
        filename="talk.mp3",
        start_time=65.0,
    )
    # 65s = 01:05, 0 = 00:00 (end_time defaults to 0.0)
    assert c.format() == "[4] talk.mp3 @ 01:05–00:00"

def test_citation_format_pdf_no_page_or_section() -> None:
    c = Citation(
        number=5,
        source_type="pdf",
        filepath="/docs/file.pdf",
        filename="file.pdf",
    )
    assert c.format() == "[5] file.pdf"
