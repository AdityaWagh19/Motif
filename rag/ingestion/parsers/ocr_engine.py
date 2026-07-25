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
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "easyocr is not installed. Run: pip install easyocr"
            ) from exc

        log.info("Initialising EasyOCR...")
        # Since EasyOCR needs to download models initially, we use English only
        _ocr_instance = easyocr.Reader(['en'])
        
    return _ocr_instance
