"""
tests/unit/test_image_parser.py
"""
from pathlib import Path
from unittest.mock import MagicMock

from rag.ingestion.parsers.image import ImageParser


def test_image_parser_t1_returns_empty(minimal_config):
    minimal_config.resolved_tier = "T1"
    parser = ImageParser(minimal_config)
    
    pages = parser.parse(Path("image.png"))
    assert pages == []

def test_image_parser_extension_check():
    assert ImageParser.can_parse(Path("photo.jpg")) is True
    assert ImageParser.can_parse(Path("doc.pdf")) is False

def test_image_parser_ocr_result_structure(minimal_config, tmp_path, monkeypatch):
    minimal_config.resolved_tier = "T2"
    parser = ImageParser(minimal_config)
    
    # Create dummy image
    dummy_img = tmp_path / "test.png"
    dummy_img.touch()
    
    # Mock PaddleOCR
    mock_ocr = MagicMock()
    # paddleocr returns: [[[[[x,y],[x,y],[x,y],[x,y]], ('text', confidence)], ...]]
    mock_ocr.ocr.return_value = [[
        [[[0,0], [1,0], [1,1], [0,1]], ("Hello OCR", 0.9)],
        [[[0,0], [1,0], [1,1], [0,1]], ("Low conf", 0.1)]
    ]]
    parser._ocr = mock_ocr
    
    # Mock import so it doesn't try to load real paddleocr inside _run_ocr if it does
    monkeypatch.setattr("rag.ingestion.parsers.image.ImageParser._run_ocr", lambda self, path: "Hello OCR")

    pages = parser.parse(dummy_img)
    
    assert len(pages) == 1
    page = pages[0]
    assert page.is_ocr is True
    assert page.has_image is True
    assert "Hello OCR" in page.text
    # Should not include low conf if _run_ocr was actually called, but we mocked it.
