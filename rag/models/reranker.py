"""
rag/models/reranker.py — Cross-encoder ONNX inference wrapper.

Full Phase 3 implementation (replaces Phase 2 no-op stub).

Models:
  T1/T2: cross-encoder/ms-marco-MiniLM-L-12-v2  (134 MB)
  T3:    BAAI/bge-reranker-base                  (280 MB)

Model directory structure:
    models/MiniLM-L12-v2/
        tokenizer.json
        tokenizer_config.json
        model_quantized.onnx   ← preferred (INT8)
            OR
        model.onnx             ← fallback (FP32)

Algorithm:
  1. For each passage: encode pair [query, passage] with [CLS] q [SEP] p [SEP]
  2. Run ONNX inference → logits of shape (batch, 1) or (batch, 2)
  3. If (batch, 2): softmax and take column-1 (relevant class probability)
     If (batch, 1): raw logit (higher = more relevant)
  4. Return float32 array of shape (len(passages),)

Always accessed through ModelManager.get_reranker() — never instantiate directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Maximum sequence length accepted by MiniLM-L12-v2 and bge-reranker-base.
MAX_SEQ_LEN = 512

# Batch size for ONNX inference (balances OOM risk vs. throughput).
_INFERENCE_BATCH = 16


class Reranker:
    """
    ONNX cross-encoder for passage reranking.

    Scores (query, passage) pairs. Higher score → more relevant.
    """

    def __init__(self, model_dir: Path) -> None:
        """
        Initialise paths. Does NOT load the model — call _load() first.

        Args:
            model_dir: Directory containing tokenizer.json and model.onnx
                       (or model_quantized.onnx).

        Raises:
            FileNotFoundError: Deferred to _load() if files are missing.
        """
        self._model_dir = model_dir
        self._session: Optional[object] = None   # ort.InferenceSession
        self._tokenizer: Optional[object] = None  # tokenizers.Tokenizer

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """
        Load the ONNX session and HuggingFace tokenizer.

        Prefers model_quantized.onnx (INT8) over model.onnx (FP32) for
        lower RAM and faster CPU inference.

        Raises:
            FileNotFoundError: If no ONNX model or tokenizer.json is found.
            RuntimeError:      If ONNX Runtime fails to load the model.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is not installed. Run: pip install onnxruntime"
            ) from exc

        # Suppress ONNX runtime warnings/errors about missing CUDA dependencies etc.
        ort.set_default_logger_severity(4)

        try:
            from tokenizers import Tokenizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "tokenizers is not installed. Run: pip install tokenizers"
            ) from exc

        # Find ONNX file — quantized preferred
        onnx_path: Optional[Path] = None
        for name in (
            "onnx/model_O3.onnx", 
            "onnx/model_quantized.onnx", 
            "onnx/model.onnx", 
            "model_quantized.onnx", 
            "model.onnx"
        ):
            candidate = self._model_dir / name
            if candidate.exists():
                onnx_path = candidate
                break

        if onnx_path is None:
            raise FileNotFoundError(
                f"No ONNX model found in {self._model_dir}. "
                "Expected model_quantized.onnx or model.onnx. "
                "Run `motif setup` to download models."
            )

        tok_path = self._model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tok_path}. "
                "Run `motif setup` to download models."
            )

        log.info("Loading reranker ONNX from %s", onnx_path)

        sess_opts = ort.SessionOptions()  # type: ignore[attr-defined]
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore[attr-defined]
        )
        self._session = ort.InferenceSession(  # type: ignore[attr-defined]
            str(onnx_path),
            sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        tokenizer = Tokenizer.from_file(str(tok_path))  # type: ignore[attr-defined]
        tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer = tokenizer

        log.info("Reranker loaded — vocab size %d", tokenizer.get_vocab_size())

    def unload(self) -> None:
        """Release ONNX session memory."""
        self._session = None
        self._tokenizer = None

    def is_loaded(self) -> bool:
        """Return True if the ONNX session is ready for inference."""
        return self._session is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(self, query: str, passages: List[str]) -> np.ndarray:
        """
        Score a list of passages against a query.

        Args:
            query:    The user's query string.
            passages: List of passage texts to score.

        Returns:
            float32 numpy array of shape (len(passages),).
            Values are relevance scores — higher is more relevant.
            Returns empty array if passages is empty.

        Raises:
            RuntimeError: If _load() has not been called.
        """
        if self._session is None or self._tokenizer is None:
            raise RuntimeError(
                "Reranker is not loaded. Call _load() before score()."
            )

        if not passages:
            return np.array([], dtype=np.float32)

        all_scores: List[np.ndarray] = []

        for i in range(0, len(passages), _INFERENCE_BATCH):
            batch = passages[i : i + _INFERENCE_BATCH]

            # Encode as sequence pairs: tokenizer handles [CLS] q [SEP] p [SEP]
            encodings = self._tokenizer.encode_batch(  # type: ignore[union-attr]
                [[query, p] for p in batch]
            )

            input_ids = np.array(
                [e.ids for e in encodings], dtype=np.int64
            )
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
            token_type_ids = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )

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
            logits: np.ndarray = outputs[0]  # (batch, 1) or (batch, 2)

            if logits.shape[-1] == 2:
                # Binary classification — softmax and take positive class score
                exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs = exp / exp.sum(axis=-1, keepdims=True)
                batch_scores = probs[:, 1]
            else:
                # Single logit — use raw value (higher = more relevant)
                batch_scores = logits[:, 0]

            all_scores.append(batch_scores)

        return np.concatenate(all_scores).astype(np.float32)
