"""
tests/integration/test_latency.py — Integration tests for latency measurement.

These tests are marked @pytest.mark.slow and are skipped unless actual models
are present (handled by conftest.py skip_if_no_model autouse fixture).

The unit-level tests validate the latency API contract without needing real models.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.evaluation.latency_test import run_latency_test


def _make_mock_pipeline():
    """Return a MagicMock that quacks like QueryPipeline."""
    mock_result = MagicMock()
    mock_result.text = "mock answer"
    mock_result.citations = []
    mock_instance = MagicMock()
    mock_instance.answer.return_value = mock_result
    return mock_instance


def test_run_latency_test_returns_correct_keys(minimal_config):
    """run_latency_test returns a dict with all expected stat keys."""
    with patch("rag.pipeline.QueryPipeline", return_value=_make_mock_pipeline()):
        questions = ["What is AI?", "What is RAG?", "Explain transformers."] * 2
        results = run_latency_test(questions, minimal_config, warmup=2)

    assert "p50_ms" in results
    assert "p95_ms" in results
    assert "p99_ms" in results
    assert "min_ms" in results
    assert "max_ms" in results
    assert "n_queries" in results
    assert "tier" in results


def test_run_latency_test_n_queries_count(minimal_config):
    """n_queries in results equals the number of measured (non-warmup) questions."""
    with patch("rag.pipeline.QueryPipeline", return_value=_make_mock_pipeline()):
        questions = ["q1", "q2", "q3", "q4", "q5"]
        results = run_latency_test(questions, minimal_config, warmup=2)

    # 5 questions measured (warmup prepended but not counted)
    assert results["n_queries"] == 5


def test_run_latency_test_tier_matches_config(minimal_config):
    """tier in results matches config.resolved_tier."""
    with patch("rag.pipeline.QueryPipeline", return_value=_make_mock_pipeline()):
        results = run_latency_test(["q1", "q2"], minimal_config, warmup=0)

    assert results["tier"] == "T1"


def test_run_latency_test_raises_on_empty_questions(minimal_config):
    """run_latency_test raises ValueError if questions list is empty."""
    with pytest.raises(ValueError, match="must not be empty"):
        run_latency_test([], minimal_config, warmup=0)


def test_run_latency_test_p50_leq_p95(minimal_config):
    """P50 is always <= P95 <= P99 <= max."""
    with patch("rag.pipeline.QueryPipeline", return_value=_make_mock_pipeline()):
        questions = [f"question {i}" for i in range(20)]
        results = run_latency_test(questions, minimal_config, warmup=2)

    assert results["min_ms"] <= results["p50_ms"]
    assert results["p50_ms"] <= results["p95_ms"]
    assert results["p95_ms"] <= results["p99_ms"]
    assert results["p99_ms"] <= results["max_ms"]


@pytest.mark.slow
def test_latency_p95_within_target(minimal_config, sample_md):
    """T1 P95 latency target: <= 11 000 ms (TRD NFR-07)."""
    from rag.ingestion import ingest_path

    ingest_path(sample_md, config=minimal_config, recursive=False, console=None)

    questions = [
        "What does this document cover?",
        "What are the main topics?",
        "Summarise the key points.",
    ] * 4  # 12 total + 2 warmup

    results = run_latency_test(questions, config=minimal_config, warmup=2)
    assert results["p95_ms"] <= 11_000, (
        f"P95 latency {results['p95_ms']:.0f} ms exceeds T1 target of 11 000 ms"
    )
    assert results["n_queries"] == 12
