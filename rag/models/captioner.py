"""
rag/models/captioner.py - Wrapper for moondream2 image captioning model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

class Captioner:
    """moondream2 GGUF/ONNX wrapper for image captioning."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for moondream2. "
                "Run `pip install transformers torch torchvision Pillow`"
            ) from exc

        log.info("Loading moondream2 captioner from %s", self._model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir), trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._model_dir),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self._model.eval()

    def caption(self, image_path: Path) -> str:
        """Generate a one-sentence description of the image."""
        if self._model is None or self._tokenizer is None:
            self._load()

        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        enc_image = self._model.encode_image(img)  # type: ignore
        result = self._model.answer_question(enc_image, "Describe this image in one sentence.", self._tokenizer)  # type: ignore
        return result.strip()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            import gc
            gc.collect()
