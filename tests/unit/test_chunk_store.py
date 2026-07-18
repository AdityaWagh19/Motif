"""
tests/unit/test_chunk_store.py — Unit tests for rag.storage.chunk_store.ChunkStore

All tests use the minimal_config and tmp_db_root fixtures from conftest.py.
No model downloads or external services required.
"""
from __future__ import annotations

import pytest

from rag.storage.chunk_store import ChunkStore
from rag.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    id: str,
    text: str = "Sample text for testing.",
    source: str = "/docs/test.pdf",
    filename: str = "test.pdf",
    source_type: str = "pdf",
    page: int = 1,
    **kwargs,
) -> Chunk:
    """Build a minimal Chunk with sensible defaults for tests."""
    return Chunk(
        id=id,
        text=text,
        source=source,
        filename=filename,
        source_type=source_type,
        page=page,
        **kwargs,
    )


@pytest.fixture()
def store(minimal_config) -> ChunkStore:
    """Fresh ChunkStore backed by the test database root."""
    s = ChunkStore(minimal_config)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInsertAndFetch:
    def test_insert_and_fetch_basic(self, store):
        chunk = _make_chunk(id="abc123", text="Hello world", page=1)
        store.insert(chunk)

        result = store.fetch("abc123")
        assert result is not None
        assert result.text == "Hello world"
        assert result.page == 1
        assert result.source_type == "pdf"
        assert result.id == "abc123"

    def test_fetch_missing_returns_none(self, store):
        result = store.fetch("nonexistent_id")
        assert result is None

    def test_insert_or_replace(self, store):
        """A second insert with the same id must overwrite the first."""
        store.insert(_make_chunk(id="abc", text="Original"))
        store.insert(_make_chunk(id="abc", text="Updated"))
        result = store.fetch("abc")
        assert result is not None
        assert result.text == "Updated"

    def test_insert_batch(self, store):
        chunks = [_make_chunk(id=f"c{i}", text=f"text {i}") for i in range(10)]
        store.insert_batch(chunks)
        assert store.count() == 10

    def test_insert_batch_empty_is_noop(self, store):
        store.insert_batch([])
        assert store.count() == 0


class TestFetchBatch:
    def test_fetch_batch_returns_found_only(self, store):
        chunks = [_make_chunk(id=f"c{i}") for i in range(5)]
        store.insert_batch(chunks)

        results = store.fetch_batch(["c0", "c2", "c4", "nonexistent"])
        assert len(results) == 3
        ids = {r.id for r in results}
        assert ids == {"c0", "c2", "c4"}

    def test_fetch_batch_empty_input(self, store):
        results = store.fetch_batch([])
        assert results == []

    def test_fetch_by_source(self, store):
        store.insert(_make_chunk(id="x1", source="/docs/a.pdf"))
        store.insert(_make_chunk(id="x2", source="/docs/a.pdf"))
        store.insert(_make_chunk(id="x3", source="/docs/b.pdf"))

        result = store.fetch_by_source("/docs/a.pdf")
        assert len(result) == 2
        assert all(c.source == "/docs/a.pdf" for c in result)


class TestDeleteBySource:
    def test_delete_by_source(self, store):
        store.insert(_make_chunk(id="a", source="/docs/a.pdf"))
        store.insert(_make_chunk(id="b", source="/docs/a.pdf"))
        store.insert(_make_chunk(id="c", source="/docs/b.pdf"))

        n = store.delete_by_source("/docs/a.pdf")
        assert n == 2
        assert store.count() == 1
        assert store.fetch("c") is not None
        assert store.fetch("a") is None

    def test_delete_by_source_nonexistent(self, store):
        n = store.delete_by_source("/docs/does_not_exist.pdf")
        assert n == 0


class TestAggregates:
    def test_count(self, store):
        assert store.count() == 0
        store.insert(_make_chunk(id="1"))
        store.insert(_make_chunk(id="2"))
        assert store.count() == 2

    def test_count_documents(self, store):
        store.insert(_make_chunk(id="a", source="/docs/a.pdf"))
        store.insert(_make_chunk(id="b", source="/docs/a.pdf"))
        store.insert(_make_chunk(id="c", source="/docs/b.pdf"))
        assert store.count_documents() == 2

    def test_list_sources(self, store):
        store.insert(_make_chunk(id="1", source="/docs/z.pdf"))
        store.insert(_make_chunk(id="2", source="/docs/a.pdf"))
        sources = store.list_sources()
        assert sources == ["/docs/a.pdf", "/docs/z.pdf"]  # alphabetical

    def test_count_documents_empty(self, store):
        assert store.count_documents() == 0


class TestBoolFieldRoundtrip:
    def test_bool_fields_true(self, store):
        chunk = _make_chunk(
            id="x",
            has_table=True,
            has_image=True,
            is_ocr=True,
        )
        store.insert(chunk)
        result = store.fetch("x")
        assert result.has_table is True
        assert result.has_image is True
        assert result.is_ocr is True

    def test_bool_fields_false(self, store):
        chunk = _make_chunk(
            id="y",
            has_table=False,
            has_image=False,
            is_ocr=False,
        )
        store.insert(chunk)
        result = store.fetch("y")
        assert result.has_table is False
        assert result.has_image is False
        assert result.is_ocr is False


class TestOptionalFields:
    def test_optional_fields_none(self, store):
        chunk = _make_chunk(id="opt", page=None, section=None, language=None)
        store.insert(chunk)
        result = store.fetch("opt")
        assert result.page is None
        assert result.section is None
        assert result.language is None

    def test_optional_fields_set(self, store):
        chunk = _make_chunk(
            id="full",
            page=42,
            section="Results",
            language="en",
            start_time=10.5,
            end_time=25.3,
            content_hash="deadbeef",
            token_count=128,
            indexed_at="2026-01-01T00:00:00Z",
        )
        store.insert(chunk)
        result = store.fetch("full")
        assert result.page == 42
        assert result.section == "Results"
        assert result.language == "en"
        assert result.start_time == pytest.approx(10.5)
        assert result.end_time == pytest.approx(25.3)
        assert result.content_hash == "deadbeef"
        assert result.token_count == 128
