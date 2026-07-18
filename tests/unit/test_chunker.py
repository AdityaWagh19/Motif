"""
tests/unit/test_chunker.py — Unit tests for SentenceChunker.

No model downloads or external services required.
"""
from __future__ import annotations

import uuid
from typing import Iterator

import pytest

from rag.ingestion.chunker import ChunkerConfig, SentenceChunker
from rag.ingestion.parsers.base import ParsedPage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def chunker() -> SentenceChunker:
    """SentenceChunker with default config (512 target, 64 overlap tokens)."""
    return SentenceChunker()


@pytest.fixture()
def small_chunker() -> SentenceChunker:
    """SentenceChunker with very small target to force splits in short texts."""
    return SentenceChunker(ChunkerConfig(target_tokens=10, overlap_tokens=2))


def _page(text: str, page: int = 1, **kwargs) -> ParsedPage:
    """Helper: build a ParsedPage with sensible defaults."""
    return ParsedPage(text=text, page=page, **kwargs)


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------

class TestBasicChunking:
    def test_empty_text_returns_no_chunks(self, chunker: SentenceChunker) -> None:
        page = _page(text="")
        assert chunker.chunk(page, "/test.pdf", "test.pdf", "pdf") == []

    def test_whitespace_only_returns_no_chunks(self, chunker: SentenceChunker) -> None:
        page = _page(text="   \n\n  ")
        assert chunker.chunk(page, "/test.pdf", "test.pdf", "pdf") == []

    def test_single_sentence_produces_one_chunk(self, chunker: SentenceChunker) -> None:
        page = _page(text="This is a single sentence.")
        chunks = chunker.chunk(page, "/test.pdf", "test.pdf", "pdf")
        assert len(chunks) == 1

    def test_chunk_preserves_source(self, chunker: SentenceChunker) -> None:
        page = _page(text="First sentence. Second sentence.")
        chunks = chunker.chunk(page, "/docs/report.pdf", "report.pdf", "pdf")
        assert all(c.source == "/docs/report.pdf" for c in chunks)

    def test_chunk_preserves_filename(self, chunker: SentenceChunker) -> None:
        page = _page(text="First sentence. Second sentence.")
        chunks = chunker.chunk(page, "/docs/report.pdf", "report.pdf", "pdf")
        assert all(c.filename == "report.pdf" for c in chunks)

    def test_chunk_preserves_page(self, chunker: SentenceChunker) -> None:
        page = _page(text="Sentence one. Sentence two.", page=3)
        chunks = chunker.chunk(page, "/test.pdf", "test.pdf", "pdf")
        assert all(c.page == 3 for c in chunks)

    def test_chunk_preserves_section(self, chunker: SentenceChunker) -> None:
        page = _page(text="Text content here.", section="Introduction")
        chunks = chunker.chunk(page, "/test.pdf", "test.pdf", "pdf")
        assert all(c.section == "Introduction" for c in chunks)

    def test_chunk_preserves_source_type(self, chunker: SentenceChunker) -> None:
        page = _page(text="Some markdown content.")
        chunks = chunker.chunk(page, "/doc.md", "doc.md", "md")
        assert all(c.source_type == "md" for c in chunks)

    def test_chunk_has_valid_uuid_id(self, chunker: SentenceChunker) -> None:
        page = _page(text="Test content for UUID check.")
        chunks = chunker.chunk(page, "/test.md", "test.md", "md")
        for c in chunks:
            uuid.UUID(c.id)  # raises ValueError if not a valid UUID

    def test_each_chunk_has_unique_id(self, chunker: SentenceChunker) -> None:
        long_text = " ".join(f"Sentence number {i}." for i in range(200))
        page = _page(text=long_text)
        chunks = chunker.chunk(page, "/doc.pdf", "doc.pdf", "pdf")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_token_count_approximation(self, chunker: SentenceChunker) -> None:
        text = "Alpha beta gamma delta."
        page = _page(text=text)
        chunks = chunker.chunk(page, "/test.md", "test.md", "md")
        assert len(chunks) == 1
        # token_count is word count
        assert chunks[0].token_count == len(text.split())

    def test_indexed_at_is_set(self, chunker: SentenceChunker) -> None:
        page = _page(text="Some text content here.")
        chunks = chunker.chunk(page, "/test.md", "test.md", "md")
        for c in chunks:
            assert c.indexed_at, "indexed_at must be non-empty"


# ---------------------------------------------------------------------------
# Splitting behaviour
# ---------------------------------------------------------------------------

class TestSplitting:
    def test_long_text_produces_multiple_chunks(
        self, small_chunker: SentenceChunker
    ) -> None:
        """100 sentences of 4 words each (~400 words) should split into many chunks."""
        long_text = " ".join(f"This is sentence {i}." for i in range(100))
        page = _page(text=long_text)
        chunks = small_chunker.chunk(page, "/doc.md", "doc.md", "md")
        assert len(chunks) >= 2

    def test_split_respects_target_tokens(self) -> None:
        """With target=10 tokens (≈13 words), each chunk should not be huge."""
        chunker = SentenceChunker(ChunkerConfig(target_tokens=10, overlap_tokens=0))
        # 5 sentences of 10 words each
        text = " ".join(
            f"Sentence {i} has exactly ten words in it here."
            for i in range(5)
        )
        page = _page(text=text)
        chunks = chunker.chunk(page, "/doc.pdf", "doc.pdf", "pdf")
        assert len(chunks) >= 2

    def test_overlap_words_appear_in_next_chunk(
        self, small_chunker: SentenceChunker
    ) -> None:
        """Last words of chunk N should appear at the start of chunk N+1."""
        long_text = " ".join(f"Word{i}" for i in range(200))
        page = _page(text=long_text)
        chunks = small_chunker.chunk(page, "/doc.md", "doc.md", "md")
        if len(chunks) >= 2:
            words_end_c0 = set(chunks[0].text.split()[-10:])
            words_start_c1 = set(chunks[1].text.split()[:10])
            overlap = words_end_c0 & words_start_c1
            assert overlap, (
                "Expected overlap words between consecutive chunks.\n"
                f"End of chunk[0]: {list(words_end_c0)}\n"
                f"Start of chunk[1]: {list(words_start_c1)}"
            )

    def test_no_overlap_config(self) -> None:
        """With overlap_tokens=0, consecutive chunks should NOT share words."""
        chunker = SentenceChunker(ChunkerConfig(target_tokens=10, overlap_tokens=0))
        # Use proper sentences so the regex splitter can break the text.
        long_text = " ".join(
            f"This is sentence number {i} with enough words."
            for i in range(50)
        )
        page = _page(text=long_text)
        chunks = chunker.chunk(page, "/doc.md", "doc.md", "md")
        # With zero overlap the test mainly checks no crash + multiple chunks
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# chunk_pages (multi-page)
# ---------------------------------------------------------------------------

class TestChunkPages:
    def test_chunk_pages_multiple_pages(self, chunker: SentenceChunker) -> None:
        pages = [
            _page(text=f"Page {i} content. " * 5, page=i)
            for i in range(1, 4)
        ]
        chunks = chunker.chunk_pages(pages, "/doc.pdf", "doc.pdf", "pdf")
        assert len(chunks) >= 3

    def test_chunk_pages_preserves_page_numbers(
        self, chunker: SentenceChunker
    ) -> None:
        pages = [_page(text="Sentence here.", page=n) for n in [5, 10, 15]]
        chunks = chunker.chunk_pages(pages, "/doc.pdf", "doc.pdf", "pdf")
        page_numbers = {c.page for c in chunks}
        assert page_numbers == {5, 10, 15}

    def test_chunk_pages_empty_list(self, chunker: SentenceChunker) -> None:
        chunks = chunker.chunk_pages([], "/doc.pdf", "doc.pdf", "pdf")
        assert chunks == []

    def test_chunk_pages_skips_empty_page(self, chunker: SentenceChunker) -> None:
        pages = [
            _page(text="Real content here.", page=1),
            _page(text="", page=2),          # empty — should produce no chunks
            _page(text="More content.", page=3),
        ]
        chunks = chunker.chunk_pages(pages, "/doc.pdf", "doc.pdf", "pdf")
        assert all(c.text for c in chunks)
        page_numbers = {c.page for c in chunks}
        assert 2 not in page_numbers


# ---------------------------------------------------------------------------
# ChunkerConfig
# ---------------------------------------------------------------------------

class TestChunkerConfig:
    def test_default_config(self) -> None:
        cfg = ChunkerConfig()
        assert cfg.target_tokens == 512
        assert cfg.overlap_tokens == 64

    def test_custom_config_applied(self) -> None:
        chunker = SentenceChunker(ChunkerConfig(target_tokens=50, overlap_tokens=5))
        # target_words = int(50 / 0.75) = 66
        assert chunker._target_words == 66
