"""
rag/models/embedder.py — nomic-embed-text-v1.5 ONNX INT8 inference wrapper.

Used by:
  - rag.ingestion   (via ModelManager): encode chunks for Qdrant upsert
  - rag.retrieval   (via ModelManager): encode queries + HyDE hypotheticals

Always accessed through ModelManager.get_embedder() — never instantiated directly.

Phase 0: Class skeleton with interface defined. Implementation in Phase 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Embedder:
    """
    Wrapper around the nomic-embed-text-v1.5 ONNX INT8 model.

    Produces 768-dimensional float32 embeddings.
    The model directory must contain: onnx/model_quantized.onnx and tokenizer files.
    """

    EMBEDDING_DIM = 768

    def __init__(self, model_dir: Path) -> None:
        """
        Load the ONNX session and tokenizer.

        Args:
            model_dir: Path to the nomic-embed-text-v1.5 ONNX model directory.

        Raises:
            FileNotFoundError: If the model directory or ONNX file is missing.
            RuntimeError:      If ONNX Runtime cannot load the model.
        """
        self._model_dir = model_dir
        self._session = None      # onnxruntime.InferenceSession — set in _load()
        self._tokenizer = None    # tokenizers.Tokenizer — set in _load()
        # Phase 1: self._load()

    def encode(self, text: str, prefix: str = "search_query: ") -> "np.ndarray":
        """
        Encode a single text string into a unit-normalised embedding vector.

        Args:
            text:   The text to embed.
            prefix: nomic-embed task prefix.
                    "search_query: " for queries.
                    "search_document: " for chunks at indexing time.

        Returns:
            float32 numpy array of shape (768,), L2-normalised.
        """
        raise NotImplementedError("Embedder.encode() — implemented in Phase 1")

    def encode_batch(
        self,
        texts: list[str],
        prefix: str = "search_document: ",
        batch_size: int = 32,
    ) -> "np.ndarray":
        """
        Encode a list of texts in batches.

        Args:
            texts:      List of strings to embed.
            prefix:     Task prefix applied to all texts.
            batch_size: Number of texts per ONNX inference call.

        Returns:
            float32 numpy array of shape (len(texts), 768), each row L2-normalised.
        """
        raise NotImplementedError("Embedder.encode_batch() — implemented in Phase 1")

    def is_loaded(self) -> bool:
        """Return True if the ONNX session is loaded and ready."""
        return self._session is not None

    def unload(self) -> None:
        """Release ONNX session memory. ModelManager calls this after ingestion on T1."""
        self._session = None
        self._tokenizer = None
