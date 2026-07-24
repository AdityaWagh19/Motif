"""
rag/ingestion/parsers/ocr_engine.py — Shared OCR engine singleton.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_ocr_instance = None


def get_ocr():
    """
    Returns a shared PaddleOCR instance.
    Initializes the model on the first call.
    """
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Run: pip install paddleocr"
            ) from exc

        log.info("Initialising PaddleOCR...")
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang="en")
        
    return _ocr_instance
