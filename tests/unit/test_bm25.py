"""
tests/unit/test_bm25.py — Unit tests for rag.retrieval.bm25_index.BM25Index

Covers: add, add_batch, search, delete, delete_by_source, count,
        persist/reload, duplicate handling, zero-score filtering.

No model downloads or external services required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.retrieval.bm25_index import BM25Index
from rag.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(id: str, text: str, source: str = "/docs/test.md") -> Chunk:
    """Build a minimal Chunk for BM25 tests."""
    return Chunk(
        id=id,
        text=text,
        source=source,
        filename="test.md",
        source_type="md",
    )


@pytest.fixture()
def index(minimal_config) -> BM25Index:
    """Fresh BM25Index backed by the test db root (no existing pickle file)."""
    # Ensure no stale index from a previous test run
    idx_path = minimal_config.db_root / "bm25" / "index.pkl"
    if idx_path.exists():
        idx_path.unlink()
    return BM25Index(minimal_config)


# ---------------------------------------------------------------------------
# Tests: basic add and search
# ---------------------------------------------------------------------------

class TestAddAndSearch:
    def test_add_and_search_basic(self, index):
        chunks = [
            _make_chunk("1", "the cat sat on the mat"),
            _make_chunk("2", "the dog barked loudly at night"),
            _make_chunk("3", "neural networks for natural language processing"),
        ]
        for c in chunks:
            index.add(c)

        results = index.search("cat mat", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "1"  # most relevant first

    def test_search_returns_correct_top_k(self, index):
        for i in range(20):
            index.add(_make_chunk(str(i), f"document number {i} about unique token tok{i}"))

        results = index.search("unique token", top_k=5)
        assert len(results) <= 5

    def test_search_empty_index_returns_empty(self, index):
        results = index.search("anything at all", top_k=10)
        assert results == []

    def test_search_empty_query_returns_empty(self, index):
        index.add(_make_chunk("1", "some text here"))
        results = index.search("", top_k=10)
        assert results == []

    def test_search_results_sorted_descending(self, index):
        index.add(_make_chunk("1", "cat cat cat"))
        index.add(_make_chunk("2", "cat dog"))
        index.add(_make_chunk("3", "dog dog dog"))

        results = index.search("cat", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_limits_results(self, index):
        for i in range(10):
            index.add(_make_chunk(str(i), f"common word test document {i}"))

        results = index.search("common word test", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Tests: zero-score filtering
# ---------------------------------------------------------------------------

class TestZeroScoreFilter:
    def test_zero_score_results_filtered(self, index):
        """Terms with no overlap should produce no results."""
        index.add(_make_chunk("1", "cats and dogs are pets"))
        results = index.search("quantum mechanics string theory", top_k=10)
        # Either empty or all scores > 0
        for _, score in results:
            assert score > 0.0

    def test_non_matching_query_returns_empty(self, index):
        """Verify empty return when query has no overlap with corpus."""
        index.add(_make_chunk("1", "the quick brown fox"))
        index.add(_make_chunk("2", "jumped over the lazy dog"))
        results = index.search("xyz123 nonexistent token", top_k=10)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_empty(self, index):
        assert index.count() == 0

    def test_count_after_adds(self, index):
        for i in range(7):
            index.add(_make_chunk(str(i), f"text document {i}"))
        assert index.count() == 7

    def test_count_after_delete(self, index):
        index.add(_make_chunk("a", "text a"))
        index.add(_make_chunk("b", "text b"))
        index.delete("a")
        assert index.count() == 1


# ---------------------------------------------------------------------------
# Tests: delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_from_search(self, index):
        index.add(_make_chunk("keep", "keep this document forever"))
        index.add(_make_chunk("remove", "remove this document please"))

        index.delete("remove")

        results = index.search("remove document please", top_k=5)
        ids = [r[0] for r in results]
        assert "remove" not in ids

    def test_delete_returns_true_on_success(self, index):
        index.add(_make_chunk("x", "some text"))
        assert index.delete("x") is True

    def test_delete_returns_false_when_not_found(self, index):
        assert index.delete("nonexistent_id") is False

    def test_delete_reduces_count(self, index):
        index.add(_make_chunk("a", "text"))
        index.add(_make_chunk("b", "text"))
        index.delete("a")
        assert index.count() == 1


# ---------------------------------------------------------------------------
# Tests: duplicate handling
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_add_duplicate_replaces_not_appends(self, index):
        """Adding a chunk with an existing id must replace it, not create a second entry."""
        index.add(_make_chunk("dup", "original text about animals"))
        index.add(_make_chunk("dup", "replacement text about mathematics"))
        assert index.count() == 1

    def test_add_duplicate_updates_search(self, index):
        index.add(_make_chunk("dup", "original text about animals cats dogs"))
        index.add(_make_chunk("dup", "replacement text about quantum mathematics"))

        # Old content should NOT dominate search
        results_new = index.search("quantum mathematics", top_k=1)
        results_old = index.search("animals cats dogs", top_k=1)

        # After replacement, "animals cats dogs" should return nothing for "dup"
        # because the text was replaced
        ids_old = [r[0] for r in results_old]
        ids_new = [r[0] for r in results_new]

        # The replacement text should be searchable
        assert "dup" in ids_new
        # The old text should not be found (overwritten)
        assert "dup" not in ids_old

    def test_add_batch_handles_duplicates(self, index):
        index.add(_make_chunk("x", "first version of text"))
        chunks = [
            _make_chunk("x", "second version completely different"),
            _make_chunk("y", "another document here"),
        ]
        index.add_batch(chunks)
        assert index.count() == 2  # "x" replaced + "y" added — not 3


# ---------------------------------------------------------------------------
# Tests: add_batch
# ---------------------------------------------------------------------------

class TestAddBatch:
    def test_add_batch_adds_all(self, index):
        chunks = [_make_chunk(str(i), f"batch document {i}") for i in range(5)]
        index.add_batch(chunks)
        assert index.count() == 5

    def test_add_batch_empty_is_noop(self, index):
        index.add_batch([])
        assert index.count() == 0

    def test_add_batch_searchable(self, index):
        # N=3 ensures IDF = log((3-1+0.5)/(1+0.5)) ≈ 0.51 > 0 for unique terms.
        # With N≤2 BM25Okapi IDF becomes 0 or negative and scores would be 0,
        # causing the match to be filtered out by the == 0.0 filter.
        chunks = [
            _make_chunk("ocean", "the deep blue ocean with waves"),
            _make_chunk("mountain", "high mountain peaks and snow"),
            _make_chunk("desert", "dry arid desert with sand dunes"),  # third doc
        ]
        index.add_batch(chunks)
        results = index.search("ocean waves", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "ocean"


# ---------------------------------------------------------------------------
# Tests: delete_by_source
# ---------------------------------------------------------------------------

class TestDeleteBySource:
    def test_delete_by_source_removes_all_matching(self, index):
        index.add(_make_chunk("a1", "text from a", source="/docs/a.md"))
        index.add(_make_chunk("a2", "more text from a", source="/docs/a.md"))
        index.add(_make_chunk("b1", "text from b", source="/docs/b.md"))

        removed = index.delete_by_source("/docs/a.md", ["a1", "a2"])
        assert removed == 2
        assert index.count() == 1

    def test_delete_by_source_returns_count(self, index):
        index.add(_make_chunk("x", "text x", source="/docs/x.md"))
        removed = index.delete_by_source("/docs/x.md", ["x"])
        assert removed == 1

    def test_delete_by_source_empty_ids_is_noop(self, index):
        index.add(_make_chunk("z", "text z", source="/docs/z.md"))
        removed = index.delete_by_source("/docs/z.md", [])
        assert removed == 0
        assert index.count() == 1


# ---------------------------------------------------------------------------
# Tests: persist and reload
# ---------------------------------------------------------------------------

class TestPersistAndReload:
    def test_persist_and_reload(self, minimal_config):
        """Index saved by one instance must be loadable by a second instance."""
        # First instance: add chunks and save
        idx1 = BM25Index(minimal_config)
        idx1.add(_make_chunk("ocean_doc", "persistent document about deep oceans"))
        idx1.save()

        # Second instance: loads from disk
        idx2 = BM25Index(minimal_config)
        assert idx2.count() == 1
        results = idx2.search("oceans", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "ocean_doc"

    def test_add_batch_auto_saves(self, minimal_config):
        """add_batch() must call save() so the index persists without explicit save()."""
        idx1 = BM25Index(minimal_config)
        idx1.add_batch([_make_chunk("auto_save", "auto saved document content")])

        idx2 = BM25Index(minimal_config)
        assert idx2.count() == 1

    def test_corrupt_index_starts_fresh(self, minimal_config):
        """A corrupt pickle file must not crash — BM25Index must start fresh."""
        idx_path = minimal_config.db_root / "bm25" / "index.pkl"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_bytes(b"not a valid pickle file at all")

        # Should not raise
        idx = BM25Index(minimal_config)
        assert idx.count() == 0

    def test_save_is_noop_when_not_dirty(self, minimal_config):
        """save() must not write to disk if nothing changed."""
        idx = BM25Index(minimal_config)
        idx.add(_make_chunk("x", "text"))
        idx.save()  # sets dirty = False

        idx_path = minimal_config.db_root / "bm25" / "index.pkl"
        mtime_before = idx_path.stat().st_mtime

        idx.save()  # should be a no-op
        mtime_after = idx_path.stat().st_mtime

        assert mtime_before == mtime_after  # file not touched


# ---------------------------------------------------------------------------
# Tests: rebuild
# ---------------------------------------------------------------------------

class TestRebuild:
    def test_rebuild_preserves_data(self, index):
        index.add(_make_chunk("r1", "rebuild test document one"))
        index.add(_make_chunk("r2", "rebuild test document two"))
        index.rebuild()

        assert index.count() == 2
        results = index.search("rebuild test", top_k=2)
        ids = {r[0] for r in results}
        assert "r1" in ids and "r2" in ids
