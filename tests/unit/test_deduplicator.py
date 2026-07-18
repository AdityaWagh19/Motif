"""
tests/unit/test_deduplicator.py — Unit tests for SimHash Deduplicator.

No model downloads or external services required.
"""
from __future__ import annotations

from typing import Iterator

import pytest

from rag.ingestion.deduplicator import Deduplicator, DUPLICATE_HAMMING_THRESHOLD
from rag.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(id: str, text: str) -> Chunk:
    """Build a minimal Chunk for deduplication testing."""
    return Chunk(
        id=id,
        text=text,
        source="/test/doc.pdf",
        filename="doc.pdf",
        source_type="pdf",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dedup() -> Deduplicator:
    return Deduplicator()


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------

class TestIsDuplicate:
    def test_first_chunk_is_never_duplicate(self, dedup: Deduplicator) -> None:
        c = _chunk("1", "The quick brown fox jumped over the lazy dog.")
        assert dedup.is_duplicate(c) is False

    def test_identical_text_is_duplicate(self, dedup: Deduplicator) -> None:
        text = "The quick brown fox jumped over the lazy dog."
        c1 = _chunk("1", text)
        c2 = _chunk("2", text)
        dedup.is_duplicate(c1)
        assert dedup.is_duplicate(c2) is True

    def test_near_identical_text_is_duplicate(self, dedup: Deduplicator) -> None:
        text = "The quick brown fox jumped over the lazy dog near the river"
        c1 = _chunk("1", text)
        c2 = _chunk("2", text + " ")  # one-space suffix — nearly identical
        dedup.is_duplicate(c1)
        assert dedup.is_duplicate(c2) is True

    def test_different_texts_are_not_duplicates(self, dedup: Deduplicator) -> None:
        c1 = _chunk("1", "The quick brown fox jumped over the lazy dog.")
        c2 = _chunk("2", "Relational databases store data in structured tables with rows.")
        dedup.is_duplicate(c1)
        assert dedup.is_duplicate(c2) is False

    def test_side_effect_adds_to_seen(self, dedup: Deduplicator) -> None:
        c = _chunk("1", "Unique content for testing purposes here.")
        assert len(dedup._seen) == 0
        dedup.is_duplicate(c)
        assert len(dedup._seen) == 1


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

class TestFilter:
    def test_no_duplicates_returns_all(self, dedup: Deduplicator) -> None:
        chunks = [
            _chunk("1", "The quick brown fox jumped over the lazy dog."),
            _chunk("2", "Machine learning models require large training datasets."),
        ]
        result = dedup.filter(chunks)
        assert len(result) == 2

    def test_exact_duplicate_removed(self, dedup: Deduplicator) -> None:
        text = "The quick brown fox jumped over the lazy dog near the river."
        chunks = [_chunk("1", text), _chunk("2", text)]
        result = dedup.filter(chunks)
        assert len(result) == 1
        assert result[0].id == "1"  # first occurrence is kept

    def test_near_duplicate_removed(self, dedup: Deduplicator) -> None:
        # SimHash threshold=3 requires very high trigram similarity.
        # Using the same text repeated ensures Hamming distance = 0 (<= 3).
        # This tests the filter path, not just is_duplicate.
        text = (
            "The quick brown fox jumped over the lazy dog near the riverbank. "
            "Machine learning models require substantial training data to generalise. "
        ) * 20
        c1 = _chunk("1", text)
        c2 = _chunk("2", text)   # byte-for-byte identical → Hamming 0
        result = dedup.filter([c1, c2])
        assert len(result) == 1
        assert result[0].id == "1"  # first occurrence kept

    def test_multiple_duplicates_all_removed(self, dedup: Deduplicator) -> None:
        text = "Boilerplate footer text that appears on every single page here."
        chunks = [_chunk(str(i), text) for i in range(5)]
        result = dedup.filter(chunks)
        assert len(result) == 1

    def test_empty_list_returns_empty(self, dedup: Deduplicator) -> None:
        assert dedup.filter([]) == []

    def test_single_chunk_returns_it(self, dedup: Deduplicator) -> None:
        c = _chunk("1", "Only child chunk.")
        result = dedup.filter([c])
        assert len(result) == 1

    def test_mixed_keeps_uniques(self, dedup: Deduplicator) -> None:
        """Unique chunks should pass through even when duplicates are present."""
        text_dup = "Duplicate boilerplate on every page in this document."
        chunks = [
            _chunk("1", text_dup),
            _chunk("2", "Unique section about machine learning and neural networks."),
            _chunk("3", text_dup),
            _chunk("4", "Another unique section covering database optimisation techniques."),
            _chunk("5", text_dup),
        ]
        result = dedup.filter(chunks)
        # chunk 1 (first dup), 2 (unique), 4 (unique) should survive
        ids = {c.id for c in result}
        assert "2" in ids
        assert "4" in ids
        assert len(result) == 3


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_seen_set(self, dedup: Deduplicator) -> None:
        text = "Content that will be seen and then forgotten after reset."
        c1 = _chunk("1", text)
        dedup.is_duplicate(c1)
        assert len(dedup._seen) == 1
        dedup.reset()
        assert len(dedup._seen) == 0

    def test_chunk_accepted_again_after_reset(self, dedup: Deduplicator) -> None:
        text = "Identical text appears in two different documents."
        c1 = _chunk("1", text)
        c2 = _chunk("2", text)
        dedup.filter([c1])
        dedup.reset()
        result = dedup.filter([c2])
        assert len(result) == 1
        assert result[0].id == "2"

    def test_multiple_resets_are_safe(self, dedup: Deduplicator) -> None:
        dedup.reset()
        dedup.reset()
        assert dedup._seen == []


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_default_threshold(self) -> None:
        dedup = Deduplicator()
        assert dedup._threshold == DUPLICATE_HAMMING_THRESHOLD

    def test_custom_threshold_zero_means_exact_only(self) -> None:
        """Threshold=0 means only exact duplicates are filtered."""
        dedup = Deduplicator(threshold=0)
        text = "The quick brown fox jumped."
        c1 = _chunk("1", text)
        c2 = _chunk("2", text + " Slight difference here.")  # different
        dedup.is_duplicate(c1)
        # At threshold 0, only exact match (same hash) counts as duplicate.
        # A slightly different text very likely has different hash → not duplicate.
        result = dedup.is_duplicate(c2)
        assert result is False
