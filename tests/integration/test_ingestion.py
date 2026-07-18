"""
tests/integration/test_ingestion.py — End-to-end ingestion pipeline tests.

@pytest.mark.slow — requires nomic-embed-text-v1.5 ONNX model.

These tests exercise the full ingest_path() → ChunkStore + BM25 + Qdrant pipeline.
They use the minimal_config and sample_md fixtures from conftest.py.

Run with model present:
    pytest tests/integration/test_ingestion.py -v -m slow

Skip in CI (model not downloaded):
    pytest tests/ -m "not slow"
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion import ingest_path, remove_document
from rag.retrieval.bm25_index import BM25Index
from rag.retrieval.vector_store import VectorStore
from rag.storage.chunk_store import ChunkStore
from rag.storage.ingestion_tracker import IngestionTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def skip_if_no_model(minimal_config) -> None:
    """
    Skip all tests in this module when the embedding model is not downloaded.
    Marked autouse so it applies to every test automatically.
    """
    model_path = Path(minimal_config.models.embed_model)
    if not model_path.is_absolute():
        model_path = model_path.resolve()
    if not model_path.exists():
        pytest.skip(
            f"Embedding model not found at {model_path}. "
            "Run `motif setup` to download models before running integration tests."
        )


# ---------------------------------------------------------------------------
# Basic ingestion
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIngestMarkdown:
    def test_ingest_markdown_files_processed(
        self, minimal_config, sample_md: Path
    ) -> None:
        result = ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        assert result.files_processed == 1
        assert result.files_skipped == 0
        assert not result.errors

    def test_ingest_markdown_chunks_added(
        self, minimal_config, sample_md: Path
    ) -> None:
        result = ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        assert result.chunks_added >= 1

    def test_ingest_markdown_stored_in_chunk_store(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        store = ChunkStore(minimal_config)
        assert store.count() >= 1
        assert store.count_documents() == 1

    def test_ingest_markdown_stored_in_bm25(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        bm25 = BM25Index(minimal_config)
        assert bm25.count() >= 1

    def test_ingest_markdown_stored_in_vector_store(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        vs = VectorStore(minimal_config)
        assert vs.count() >= 1

    def test_ingest_markdown_tracker_records_file(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        tracker = IngestionTracker(minimal_config)
        assert tracker.is_indexed(sample_md)
        assert tracker.get_hash(sample_md) is not None


# ---------------------------------------------------------------------------
# Idempotency (skip unchanged files)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIdempotency:
    def test_second_ingest_skips_unchanged(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        result2 = ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        assert result2.files_skipped == 1
        assert result2.chunks_added == 0
        assert result2.files_processed == 0

    def test_second_ingest_does_not_duplicate_chunks(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        count_after_first = ChunkStore(minimal_config).count()

        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        count_after_second = ChunkStore(minimal_config).count()

        assert count_after_first == count_after_second


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRemoveDocument:
    def test_remove_reduces_chunk_count_to_zero(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        store = ChunkStore(minimal_config)
        assert store.count() >= 1

        n = remove_document(sample_md, config=minimal_config)
        assert n >= 1
        assert store.count() == 0

    def test_remove_clears_tracker_entry(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        remove_document(sample_md, config=minimal_config)
        tracker = IngestionTracker(minimal_config)
        assert not tracker.is_indexed(sample_md)

    def test_remove_allows_reingest(
        self, minimal_config, sample_md: Path
    ) -> None:
        ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        remove_document(sample_md, config=minimal_config)
        result = ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
        assert result.files_processed == 1
        assert result.chunks_added >= 1

    def test_remove_nonexistent_returns_zero(
        self, minimal_config, tmp_path: Path
    ) -> None:
        fake = tmp_path / "ghost.md"
        # File never ingested — remove should return 0 gracefully
        n = remove_document(fake, config=minimal_config)
        assert n == 0


# ---------------------------------------------------------------------------
# Directory ingestion
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDirectoryIngestion:
    def test_ingest_directory_non_recursive(
        self, minimal_config, tmp_path: Path
    ) -> None:
        # Create 2 markdown files in root, 1 in subdir
        (tmp_path / "a.md").write_text("# A\n\nContent of document A.", encoding="utf-8")
        (tmp_path / "b.md").write_text("# B\n\nContent of document B.", encoding="utf-8")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.md").write_text("# C\n\nContent of document C.", encoding="utf-8")

        result = ingest_path(tmp_path, config=minimal_config, recursive=False, console=None)
        # Only root files should be processed (non-recursive)
        assert result.files_processed == 2

    def test_ingest_directory_recursive(
        self, minimal_config, tmp_path: Path
    ) -> None:
        (tmp_path / "a.md").write_text("# A\n\nContent of document A.", encoding="utf-8")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "b.md").write_text("# B\n\nContent of document B.", encoding="utf-8")

        result = ingest_path(tmp_path, config=minimal_config, recursive=True, console=None)
        assert result.files_processed == 2

    def test_ingest_ignores_unsupported_types(
        self, minimal_config, tmp_path: Path
    ) -> None:
        (tmp_path / "a.md").write_text("# A\n\nContent A.", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.csv").write_text("col1,col2\n1,2", encoding="utf-8")

        result = ingest_path(tmp_path, config=minimal_config, recursive=False, console=None)
        # Only .md file should be processed
        assert result.files_processed == 1
