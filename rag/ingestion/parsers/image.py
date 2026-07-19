"""
rag/ingestion/parsers/image.py — Parser for images (OCR + captioning).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

from rag.ingestion.parsers.base import BaseParser, ParsedPage
from rag.models.model_manager import get_model_manager

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

LOW_TEXT_THRESHOLD = 50  # chars — below this, try captioning

class ImageParser(BaseParser):
    """
    Parser for image files (.png, .jpg, etc).
    Uses PaddleOCR for text extraction. 
    If use_moondream is enabled and text is low, generates a caption.
    Requires T2 or T3 config.
    """
    
    SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

    def __init__(self, config: "RAGConfig") -> None:
        self._config = config
        self._ocr = None

    def parse(self, path: Path) -> List[ParsedPage]:
        if self._config.resolved_tier == "T1":
            raise ValueError(
                "Image parsing requires T2 or T3 (PaddleOCR needs GPU or sufficient RAM). "
                "T1 (CPU-only) does not support image ingestion."
            )

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        text = self._run_ocr(path)
        caption = ""

        # Default to false if parsers config is missing
        use_moondream = getattr(self._config.parsers, "use_moondream", False)

        if use_moondream and len(text) < LOW_TEXT_THRESHOLD:
            caption = self._run_caption(path)

        if caption and text:
            full_text = f"Image caption: {caption}\n\nOCR text: {text}"
        elif caption:
            full_text = f"Image caption: {caption}"
        else:
            full_text = text

        if not full_text.strip():
            return []

        return [ParsedPage(
            text=full_text.strip(),
            is_ocr=True,
            has_image=True,
        )]

    def _run_ocr(self, path: Path) -> str:
        """Run PaddleOCR on the image. Returns extracted text."""
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Run: pip install paddleocr"
            ) from exc

        # Lazy-load PaddleOCR (slow first-time init — downloads models if missing)
        if self._ocr is None:
            log.info("Initialising PaddleOCR...")
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en")
            
        result = self._ocr.predict(str(path))
        if not result or not result[0]:
            return ""
            
        lines = []
        for line in result[0]:
            text_confidence = line[1]
            text = text_confidence[0]
            confidence = text_confidence[1]
            if confidence >= 0.6:  # drop low-confidence OCR lines
                lines.append(text)
                
        return " ".join(lines)

    def _run_caption(self, path: Path) -> str:
        """Run moondream2 to generate an image description."""
        try:
            captioner = get_model_manager().get_captioner(self._config)
            return captioner.caption(path)
        except Exception as e:
            log.warning("Image captioning failed: %s", e)
            return ""
