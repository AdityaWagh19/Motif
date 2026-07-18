"""
tests/integration/test_query.py — End-to-end query pipeline tests.

These tests require the LLM, Embedder, and Reranker models to be loaded.
Run with: pytest tests/integration/test_query.py -v -m slow
"""
from __future__ import annotations

import pytest

from rag.pipeline import QueryPipeline
from rag.ingestion import ingest_path
from rag.storage.chunk_store import ChunkStore


@pytest.mark.slow
def test_end_to_end_ingest_and_query(minimal_config, sample_md) -> None:
    # 1. Ingest the sample document
    ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    assert ChunkStore(minimal_config).count() >= 1

    # 2. Run an on-topic query
    pipeline = QueryPipeline(minimal_config)
    result = pipeline.answer(
        query="What does this document discuss?",
        history=[],
    )
    
    assert result.text
    assert len(result.text) > 20
    # Should not be a hallucination refusal for a query about the doc
    assert "cannot find" not in result.text.lower()
    assert result.passages_used > 0
    assert len(result.citations) > 0


@pytest.mark.slow
def test_query_unanswerable_returns_refusal(minimal_config, sample_md) -> None:
    # 1. Ingest the sample document
    ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    
    # 2. Run an off-topic query (e.g., about chemistry)
    pipeline = QueryPipeline(minimal_config)
    result = pipeline.answer(
        query="What is the molecular weight of caffeine?",
        history=[],
    )
    
    # Either it refuses explicitly, or it finds no relevant passages at all
    # due to the relevance threshold filtering them out.
    assert "cannot find" in result.text.lower() or len(result.citations) == 0


@pytest.mark.slow
def test_history_followup_is_injected(minimal_config, sample_md) -> None:
    ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    
    pipeline = QueryPipeline(minimal_config)
    
    # Mock history that provides explicit context
    history = [
        {"role": "user", "content": "What is the document about?"},
        {"role": "assistant", "content": "The document discusses Markdown features like bold text."},
    ]
    
    result = pipeline.answer(
        query="Expand on that feature.",
        history=history,
    )
    
    assert result.text
    assert len(result.text) > 20
    # It should understand "that feature" refers to bold text / Markdown features.
    assert "cannot find" not in result.text.lower()
