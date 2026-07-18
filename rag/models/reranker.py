"""
rag/models/reranker.py — Cross-encoder ONNX inference wrapper.

Models:
  T1/T2: cross-encoder/ms-marco-MiniLM-L-12-v2  (134 MB)
  T3:    BAAI/bge-reranker-base                  (280 MB)

Used by:
  - rag.reranking.cross_encoder (via ModelManager): score query-passage pairs

Always accessed through ModelManager.get_reranker() — never instantiated directly.

Phase 0: Class skeleton with interface defined. Implementation in Phase 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from rag.types import ScoredPassage


class Reranker:
    """
    Wrapper around a cross-encoder ONNX model for passage reranking.

    Scores (query, passage) pairs and returns logit scores.
    Higher score = more relevant.
    """

    def __init__(self, model_dir: Path) -> None:
        """
        Load the ONNX session and tokenizer.

        Args:
            model_dir: Path to the cross-encoder ONNX model directory.

        Raises:
            FileNotFoundError: If the model directory or ONNX file is missing.
            RuntimeError:      If ONNX Runtime cannot load the model.
        """
        self._model_dir = model_dir
        self._session = None      # onnxruntime.InferenceSession — set in _load()
        self._tokenizer = None    # tokenizers.Tokenizer — set in _load()
        # Phase 1: self._load()

    def score(self, query: str, passages: list[str]) -> "np.ndarray":
        """
        Score a list of passages against a query.

        Args:
            query:    The user's query string.
            passages: List of passage texts to score.

        Returns:
            float32 numpy array of shape (len(passages),).
            Values are raw logits — higher is more relevant.
            Caller is responsible for softmax / threshold filtering if needed.
        """
        raise NotImplementedError("Reranker.score() — implemented in Phase 1")

    def is_loaded(self) -> bool:
        """Return True if the ONNX session is loaded and ready."""
        return self._session is not None

    def unload(self) -> None:
        """Release ONNX session memory."""
        self._session = None
        self._tokenizer = None
