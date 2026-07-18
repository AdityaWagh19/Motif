"""
tests/unit/test_audio_parser.py
"""
import pytest
from pathlib import Path

from rag.ingestion.parsers.audio import AudioParser
from rag.ingestion.parsers.base import ParsedPage

def test_audio_parser_extension_check():
    assert AudioParser.can_parse(Path("talk.mp3")) is True
    assert AudioParser.can_parse(Path("doc.pdf")) is False

def test_audio_parser_group_segments_basic(minimal_config):
    segments = [
        {"text": "Hello world", "start": 0.0, "end": 3.0},
        {"text": "This is a test", "start": 3.0, "end": 6.0},
    ]
    parser = AudioParser(minimal_config)
    pages = parser._group_segments(segments)
    
    assert len(pages) >= 1
    assert pages[0].start_time == 0.0
    assert pages[-1].end_time == 6.0
    assert "Hello world This is a test" in pages[0].text

def test_audio_parser_timestamps_in_page(minimal_config):
    parser = AudioParser(minimal_config)
    pages = parser._group_segments([{"text": "x", "start": 10.5, "end": 15.3}])
    
    assert pages[0].start_time == 10.5
    assert pages[0].end_time == 15.3
    assert pages[0].text == "x"
