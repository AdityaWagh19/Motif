"""
tests/unit/test_embedder.py — Unit tests for the Embedder class.

@pytest.mark.slow — all tests require the nomic-embed-text-v1.5 ONNX model.

Run with model present:
    pytest tests/unit/test_embedder.py -v -m slow

Skip in CI (model not downloaded):
    pytest tests/unit/ -m "not slow"

Model path: config.models.embed_model (default: models/nomic-embed-text-v1.5/)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from rag.models.embedder import Embedder, EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Fixture: load embedder from model directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder(minimal_config) -> Iterator[Embedder]:
    """
    Load the Embedder once for all slow tests in this module.

    Skips automatically if the model directory does not exist
    (developer machine without downloaded models).
    """
    model_path = Path(minimal_config.models.embed_model)
    if not model_path.is_absolute():
        model_path = model_path.resolve()

    if not model_path.exists():
        pytest.skip(
            f"Embedding model not found at {model_path}. "
            "Run `motif setup` to download models."
        )

    e = Embedder(model_path)
    e._load()
    yield e
    e.unload()


# ---------------------------------------------------------------------------
# Lifecycle tests (no model needed)
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_not_loaded_initially(self, minimal_config) -> None:
        model_path = Path(minimal_config.models.embed_model)
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        e = Embedder(model_path)
        assert e.is_loaded() is False

    def test_encode_raises_if_not_loaded(self, minimal_config) -> None:
        model_path = Path(minimal_config.models.embed_model)
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        e = Embedder(model_path)
        with pytest.raises(RuntimeError, match="not loaded"):
            e.encode("test text")

    def test_encode_batch_raises_if_not_loaded(self, minimal_config) -> None:
        model_path = Path(minimal_config.models.embed_model)
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        e = Embedder(model_path)
        with pytest.raises(RuntimeError, match="not loaded"):
            e.encode_batch(["test text"])

    def test_unload_is_idempotent(self, minimal_config) -> None:
        model_path = Path(minimal_config.models.embed_model)
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        e = Embedder(model_path)
        e.unload()  # unload without loading first — must not raise
        e.unload()  # second unload — must not raise

    def test_load_raises_on_missing_model_dir(self, tmp_path: Path) -> None:
        e = Embedder(tmp_path / "nonexistent_model")
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            e._load()


# ---------------------------------------------------------------------------
# Inference tests (require model)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestInference:
    def test_encode_returns_correct_shape(self, embedder: Embedder) -> None:
        import numpy as np
        vec = embedder.encode("This is a test sentence for embedding.")
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32

    def test_encode_is_unit_normalised(self, embedder: Embedder) -> None:
        import numpy as np
        vec = embedder.encode("Test sentence for normalisation check.")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"

    def test_encode_batch_correct_shape(self, embedder: Embedder) -> None:
        import numpy as np
        texts = ["First text here.", "Second text here.", "Third text here."]
        vecs = embedder.encode_batch(texts)
        assert vecs.shape == (3, EMBEDDING_DIM)
        assert vecs.dtype == np.float32

    def test_encode_batch_all_rows_normalised(self, embedder: Embedder) -> None:
        import numpy as np
        texts = ["Alpha text.", "Beta text.", "Gamma text."]
        vecs = embedder.encode_batch(texts)
        for i, row in enumerate(vecs):
            norm = float(np.linalg.norm(row))
            assert abs(norm - 1.0) < 1e-5, f"Row {i} not normalised: norm={norm}"

    def test_encode_batch_empty_returns_empty(self, embedder: Embedder) -> None:
        import numpy as np
        result = embedder.encode_batch([])
        assert result.shape == (0, EMBEDDING_DIM)

    def test_similar_texts_closer_than_dissimilar(self, embedder: Embedder) -> None:
        """Semantic similarity: cats ↔ felines > cats ↔ Python loops."""
        v_cat1 = embedder.encode("cats are domestic animals kept as pets")
        v_cat2 = embedder.encode("felines are popular pets kept at home")
        v_code = embedder.encode("for loop in Python programming language")

        sim_related = float(v_cat1 @ v_cat2)
        sim_unrelated = float(v_cat1 @ v_code)
        assert sim_related > sim_unrelated, (
            f"Expected cats↔felines ({sim_related:.4f}) > "
            f"cats↔code ({sim_unrelated:.4f})"
        )

    def test_query_vs_document_prefix(self, embedder: Embedder) -> None:
        """Query and document prefixes should produce different vectors."""
        import numpy as np
        text = "The Eiffel Tower is located in Paris France."
        v_query = embedder.encode(text, prefix="search_query: ")
        v_doc = embedder.encode(text, prefix="search_document: ")
        # They should not be identical (different prefix → different embedding)
        assert not np.allclose(v_query, v_doc, atol=1e-4)

    def test_encode_batch_large(self, embedder: Embedder) -> None:
        """Batch larger than _DEFAULT_BATCH_SIZE (32) to test mini-batch logic."""
        import numpy as np
        texts = [f"Sentence number {i} for batch testing purposes." for i in range(50)]
        vecs = embedder.encode_batch(texts, batch_size=16)
        assert vecs.shape == (50, EMBEDDING_DIM)
        # All rows should be normalised
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_is_loaded_true_after_load(self, embedder: Embedder) -> None:
        assert embedder.is_loaded() is True

    def test_is_loaded_false_after_unload(self, minimal_config) -> None:
        model_path = Path(minimal_config.models.embed_model)
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        if not model_path.exists():
            pytest.skip("Model not available.")
        e = Embedder(model_path)
        e._load()
        assert e.is_loaded() is True
        e.unload()
        assert e.is_loaded() is False
