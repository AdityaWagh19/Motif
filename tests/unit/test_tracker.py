"""
tests/unit/test_tracker.py — Unit tests for rag.storage.ingestion_tracker

Tests IngestionTracker and the compute_file_hash() module-level function.
No model downloads or external services required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from rag.storage.ingestion_tracker import IngestionTracker, compute_file_hash


@pytest.fixture()
def tracker(minimal_config) -> Iterator[IngestionTracker]:
    """Fresh IngestionTracker backed by the test database root."""
    t = IngestionTracker(minimal_config)
    yield t
    t.close()


@pytest.fixture()
def tmp_file(tmp_path: Path) -> Path:
    """A temporary file with some default content."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"initial content")
    return p


# ---------------------------------------------------------------------------
# IngestionTracker tests
# ---------------------------------------------------------------------------

class TestIsIndexed:
    def test_is_indexed_false_for_new_file(self, tracker, tmp_file):
        assert tracker.is_indexed(tmp_file) is False

    def test_is_indexed_true_after_update(self, tracker, tmp_file):
        tracker.update(tmp_file, content_hash="abc123", chunk_count=5)
        assert tracker.is_indexed(tmp_file) is True

    def test_is_indexed_false_after_remove(self, tracker, tmp_file):
        tracker.update(tmp_file, content_hash="abc123", chunk_count=5)
        tracker.remove(tmp_file)
        assert tracker.is_indexed(tmp_file) is False


class TestGetHash:
    def test_get_hash_returns_none_when_not_tracked(self, tracker, tmp_file):
        assert tracker.get_hash(tmp_file) is None

    def test_get_hash_returns_stored_hash(self, tracker, tmp_file):
        tracker.update(tmp_file, content_hash="deadbeef", chunk_count=3)
        assert tracker.get_hash(tmp_file) == "deadbeef"

    def test_get_hash_updates_on_second_call(self, tracker, tmp_file):
        tracker.update(tmp_file, content_hash="hash_v1", chunk_count=3)
        tracker.update(tmp_file, content_hash="hash_v2", chunk_count=7)
        assert tracker.get_hash(tmp_file) == "hash_v2"


class TestUpdate:
    def test_update_stores_absolute_path(self, tracker, tmp_path):
        """Filepath stored must always be the resolved absolute path."""
        f = tmp_path / "doc.md"
        f.write_bytes(b"content")
        tracker.update(f, content_hash="h", chunk_count=1)
        entries = tracker.list_all()
        assert len(entries) == 1
        assert Path(entries[0]["filepath"]).is_absolute()

    def test_update_chunk_count(self, tracker, tmp_file):
        tracker.update(tmp_file, content_hash="h", chunk_count=42)
        entries = tracker.list_all()
        assert entries[0]["chunk_count"] == 42

    def test_update_sets_indexed_at(self, tracker, tmp_file):
        tracker.update(tmp_file, content_hash="h", chunk_count=1)
        entries = tracker.list_all()
        assert entries[0]["indexed_at"]  # not empty
        # Should be ISO 8601 format
        assert "T" in entries[0]["indexed_at"]


class TestRemove:
    def test_remove_existing(self, tracker, tmp_file):
        tracker.update(tmp_file, content_hash="h", chunk_count=1)
        tracker.remove(tmp_file)
        assert tracker.is_indexed(tmp_file) is False

    def test_remove_nonexistent_is_noop(self, tracker, tmp_file):
        """Removing a path that is not tracked should not raise."""
        tracker.remove(tmp_file)  # should not raise
        assert tracker.is_indexed(tmp_file) is False


class TestListAll:
    def test_list_all_empty(self, tracker):
        assert tracker.list_all() == []

    def test_list_all_returns_all_entries(self, tracker, tmp_path):
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")

        tracker.update(f1, "hash_a", 3)
        tracker.update(f2, "hash_b", 7)

        entries = tracker.list_all()
        assert len(entries) == 2
        assert any(e["chunk_count"] == 3 for e in entries)
        assert any(e["chunk_count"] == 7 for e in entries)
        assert any(e["content_hash"] == "hash_a" for e in entries)

    def test_list_all_keys(self, tracker, tmp_file):
        tracker.update(tmp_file, "h", 1)
        entries = tracker.list_all()
        assert set(entries[0].keys()) == {"filepath", "content_hash", "indexed_at", "chunk_count"}


# ---------------------------------------------------------------------------
# compute_file_hash() tests
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_hash_is_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello motif")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2

    def test_hash_is_64_hex_chars(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello motif")
        h = compute_file_hash(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_changes_with_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"version 1")
        h1 = compute_file_hash(f)
        f.write_bytes(b"version 2")
        h2 = compute_file_hash(f)
        assert h1 != h2

    def test_hash_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        h = compute_file_hash(f)
        # SHA-256 of empty string is a known constant
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_hash_large_file(self, tmp_path):
        """Ensure streaming read works for files larger than 64 KB."""
        f = tmp_path / "large.bin"
        f.write_bytes(b"x" * (200 * 1024))  # 200 KB
        h = compute_file_hash(f)
        assert len(h) == 64
