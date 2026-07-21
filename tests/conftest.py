"""
tests/conftest.py — Shared pytest fixtures.

All integration and unit tests that need a database, sample documents, or a
minimal config import from here. No test file should set up its own fixtures
for things already provided here.

Fixture scopes:
  - tmp_db_root    (function) — fresh temp dir for each test
  - minimal_config (function) — RAGConfig pointing at tmp_db_root, T1 tier
  - sample_pdf     (session)  — reused across all tests (read-only)
  - sample_md      (session)  — reused across all tests (read-only)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import (
    ChunkingConfig,
    GenerationConfig,
    HardwareConfig,
    LLMConfig,
    ModelsConfig,
    RAGConfig,
    RetrievalConfig,
    StorageConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Database / Storage fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db_root(tmp_path: Path) -> Path:
    """
    A fresh temporary directory for each test.
    Qdrant collection, SQLite databases, and session history are written here.
    Automatically cleaned up after the test.
    """
    db_root = tmp_path / "ragdb"
    db_root.mkdir()
    return db_root


@pytest.fixture(autouse=True)
def skip_if_no_model(request, minimal_config):
    """Skip test if marked 'slow' and models are not downloaded."""
    if "slow" in request.keywords:
        embed_path = Path(minimal_config.models.embed_model)
        if not embed_path.is_absolute():
            from rag.config import _get_models_dir
            embed_path = _get_models_dir() / embed_path.name
        if not embed_path.exists():
            pytest.skip(f"Embedding model not found at {embed_path}. Run `motif setup`.")
            
        llm_path = Path(minimal_config.models.llm_path)
        if not llm_path.is_absolute():
            from rag.config import _get_models_dir
            llm_path = _get_models_dir() / llm_path.name
        if not llm_path.exists():
            pytest.skip(f"LLM model not found at {llm_path}. Run `motif setup`.")


@pytest.fixture()
def minimal_config(tmp_db_root: Path) -> RAGConfig:
    """
    A minimal RAGConfig suitable for unit and integration tests.

    - Tier: T1 (CPU only, no GPU layers)
    - DB root: tmp_db_root (isolated per test)
    - Models: not downloaded — tests that need real models are marked @pytest.mark.slow
    """
    config = RAGConfig(
        hardware=HardwareConfig(tier="T1"),
        models=ModelsConfig(
            llm_path="models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
            embed_model="models/nomic-embed-text-v1.5",
            reranker="models/MiniLM-L12-v2",
        ),
        llm=LLMConfig(n_gpu_layers=0, ctx_size=2048, max_tokens=200, threads=2),
        retrieval=RetrievalConfig(
            top_k_retrieval=10,
            top_k_rerank=3,
            query_expansion="none",
        ),
        chunking=ChunkingConfig(target_tokens=256, overlap_tokens=32, use_semantic=False),
        generation=GenerationConfig(context_max_tokens=1024, streaming=False, history_turns=2),
        storage=StorageConfig(db_path=str(tmp_db_root), query_cache_enabled=False),
    )
    config.resolved_tier = "T1"
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Sample document fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    A minimal Markdown file for testing ingestion parsers and chunkers.
    Reused across all tests in the session (read-only).
    """
    p = tmp_path_factory.mktemp("docs") / "sample.md"
    p.write_text(
        "# Introduction\n\n"
        "This is a test document for Motif RAG pipeline testing.\n\n"
        "## Methods\n\n"
        "We use reciprocal rank fusion to combine dense, sparse, and BM25 results.\n\n"
        "## Results\n\n"
        "The system achieves 85% faithfulness on the evaluation dataset.\n\n"
        "## Conclusion\n\n"
        "Offline multimodal RAG is feasible on consumer hardware with careful model selection.\n",
        encoding="utf-8",
    )
    return p  # type: ignore[return-value]


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    """
    Path to a minimal PDF for testing PDF parsing.

    If no PDF fixture exists in tests/fixtures/, this fixture is skipped.
    Add a file at tests/fixtures/sample.pdf to enable PDF tests.
    """
    fixtures = Path(__file__).parent / "fixtures"
    pdf = fixtures / "sample.pdf"
    if not pdf.exists():
        pytest.skip("No sample.pdf in tests/fixtures/ — skipping PDF tests")
    return pdf
