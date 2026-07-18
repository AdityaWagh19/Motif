"""
rag/models/embedder.py — nomic-embed-text-v1.5 ONNX INT8 inference wrapper.

Replaces the Phase 0 skeleton with a full implementation.

Model directory structure (set up by `motif setup`):
    models/nomic-embed-text-v1.5/
        onnx/model_quantized.onnx      ← quantized ONNX model
        tokenizer.json                  ← HF tokenizer
        tokenizer_config.json
        special_tokens_map.json
        config.json

Inference backend: onnxruntime (CPU by default; GPU providers added in Phase 3)
Tokenizer: tokenizers (HuggingFace fast tokenizer)

Output: 768-dimensional float32 L2-normalised vectors.

nomic-embed-text-v1.5 requires a task prefix on the input text:
  - Documents:  "search_document: <text>"
  - Queries:    "search_query: <text>"

Dependency graph position:
    embedder  →  onnxruntime  (third-party)
    embedder  →  tokenizers   (third-party)
    embedder  →  numpy        (third-party)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer  # type: ignore[import]

log = logging.getLogger(__name__)

EMBEDDING_DIM: int = 768
MAX_SEQ_LEN: int = 8192          # nomic-embed-text supports up to 8 192 tokens
_DEFAULT_BATCH_SIZE: int = 4    # chunks per inference call


class Embedder:
    """
    Wrapper around the nomic-embed-text-v1.5 ONNX INT8 model.

    Always accessed through ModelManager.get_embedder() — never instantiated
    directly by application code.

    Lifecycle:
        1. ModelManager calls __init__(model_dir).
        2. ModelManager calls _load() to initialise the ONNX session.
        3. encode() / encode_batch() are called during ingestion and retrieval.
        4. ModelManager calls unload() to free ~550 MB RAM before loading the LLM.
    """

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._session: Optional[object] = None    # ort.InferenceSession
        self._tokenizer: Optional[object] = None  # tokenizers.Tokenizer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Initialise the ONNX inference session and HuggingFace tokenizer.

        Raises:
            FileNotFoundError: If the model file or tokenizer are missing.
            RuntimeError:      If onnxruntime fails to load the session.
        """
        try:
            import numpy as _np  # noqa: F401 — ensure numpy is available
            import onnxruntime as _ort
            from tokenizers import Tokenizer as _Tokenizer  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                f"Required package not installed: {exc}. "
                "Run: pip install onnxruntime tokenizers"
            ) from exc

        onnx_path = self._model_dir / "onnx" / "model_quantized.onnx"
        tok_path = self._model_dir / "tokenizer.json"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Run `motif setup` to download the embedding model."
            )
        if not tok_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tok_path}\n"
                "Run `motif setup` to download the embedding model."
            )

        log.debug("Loading ONNX session from %s", onnx_path)
        sess_opts = _ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = (
            _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = _ort.InferenceSession(
            str(onnx_path),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )

        log.debug("Loading tokenizer from %s", tok_path)
        tok = _Tokenizer.from_file(str(tok_path))
        tok.enable_truncation(max_length=MAX_SEQ_LEN)
        # Pad to the longest sequence in the batch (length=None → dynamic padding)
        tok.enable_padding(pad_id=1, pad_token="[PAD]", length=None)
        self._tokenizer = tok

        log.info("Embedder loaded: %s (dim=%d)", self._model_dir.name, EMBEDDING_DIM)

    def is_loaded(self) -> bool:
        """Return True if the ONNX session has been initialised."""
        return self._session is not None

    def unload(self) -> None:
        """
        Release the ONNX session and tokenizer, freeing ~550 MB RAM.
        Safe to call multiple times.
        """
        self._session = None
        self._tokenizer = None
        log.debug("Embedder unloaded.")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts: List[str]) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
        """
        Tokenize a batch of texts.

        Returns:
            (input_ids, attention_mask, token_type_ids) as int64 numpy arrays of shape (B, L).
        """
        import numpy as np

        encodings = self._tokenizer.encode_batch(texts)  # type: ignore[union-attr]
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encodings], dtype=np.int64
        )
        token_type_ids = np.array(
            [e.type_ids for e in encodings], dtype=np.int64
        )
        return input_ids, attention_mask, token_type_ids

    def _mean_pool_and_normalize(
        self,
        token_embeddings: "np.ndarray",
        attention_mask: "np.ndarray",
    ) -> "np.ndarray":
        """
        Mean-pool token embeddings (masked) then L2-normalise.

        Args:
            token_embeddings: (B, L, 768) float32
            attention_mask:   (B, L) int64

        Returns:
            (B, 768) float32, each row is a unit vector.
        """
        import numpy as np

        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_emb = (token_embeddings * mask).sum(axis=1)
        count = mask.sum(axis=1).clip(min=1e-9)
        mean_emb = sum_emb / count
        norms = np.linalg.norm(mean_emb, axis=1, keepdims=True).clip(min=1e-9)
        return (mean_emb / norms).astype(np.float32)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def encode(self, text: str, prefix: str = "search_query: ") -> "np.ndarray":
        """
        Encode a single text string.

        Args:
            text:   Input text.
            prefix: Task prefix. Use "search_query: " for queries,
                    "search_document: " for indexing.

        Returns:
            (768,) float32 L2-normalised vector.
        """
        if not self.is_loaded():
            raise RuntimeError("Embedder not loaded. Call _load() first.")
        return self.encode_batch([text], prefix=prefix)[0]

    def encode_batch(
        self,
        texts: List[str],
        prefix: str = "search_document: ",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> "np.ndarray":
        """
        Encode a list of texts in mini-batches.

        Args:
            texts:      Input strings.
            prefix:     Task prefix prepended to each text before encoding.
            batch_size: Number of texts per ONNX inference call.

        Returns:
            (N, 768) float32 array; each row is a unit vector.
        """
        import numpy as np

        if not self.is_loaded():
            raise RuntimeError("Embedder not loaded. Call _load() first.")
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        prefixed = [f"{prefix}{t}" for t in texts]
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i:i + batch_size]
            input_ids, attention_mask, token_type_ids = self._tokenize(batch)

            expected_inputs = [inp.name for inp in self._session.get_inputs()]  # type: ignore[union-attr]
            feed_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in expected_inputs:
                feed_dict["token_type_ids"] = token_type_ids

            outputs = self._session.run(  # type: ignore[union-attr]
                None,
                feed_dict,
            )
            # outputs[0] shape: (B, L, 768)
            token_embeddings = outputs[0]
            embeddings = self._mean_pool_and_normalize(token_embeddings, attention_mask)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
