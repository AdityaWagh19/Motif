"""
rag/ingestion/parsers/audio.py — Parser for audio (whisper.cpp transcription).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, TYPE_CHECKING
from contextlib import contextmanager

from rag.ingestion.parsers.base import BaseParser, ParsedPage

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

@contextmanager
def suppress_c_stderr():
    """Context manager to suppress C-level stderr (e.g., from whisper.cpp)."""
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_fd = os.dup(sys.stderr.fileno())
        os.dup2(null_fd, sys.stderr.fileno())
        yield
    finally:
        try:
            os.dup2(save_fd, sys.stderr.fileno())
            os.close(null_fd)
            os.close(save_fd)
        except Exception:
            pass

TARGET_WORDS = 350  # ≈ 512 tokens at 0.75 words/token ratio

class AudioParser(BaseParser):
    """
    Parser for audio files using pywhispercpp (whisper.cpp bindings).
    Groups transcripts into ~512 token chunks.
    """
    
    SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

    def __init__(self, config: "RAGConfig") -> None:
        self._config = config

    def parse(self, path: Path) -> List[ParsedPage]:
        """
        Transcribe audio file using whisper.cpp.

        Returns one ParsedPage per whisper segment group (~30-60 seconds each).
        Each ParsedPage carries start_time and end_time from whisper timestamps.
        """
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        model_path = self._get_whisper_model_path()
        segments = self._transcribe(path, model_path)

        if not segments:
            return []

        # Group segments into ~512-token chunks by time window
        return self._group_segments(segments)

    def _get_whisper_model_path(self) -> Path:
        """
        Resolve the Whisper model path.

        Relative paths are resolved against the project root (where pyproject.toml
        lives), NOT the process CWD. This ensures audio ingestion works regardless
        of which directory the user runs `motif` from.
        """
        cfg = self._config
        whisper_model = cfg.models.whisper
        path = Path(whisper_model)
        if not path.is_absolute():
            # Anchor relative paths to the project root:
            # rag/ingestion/parsers/audio.py → rag/ingestion/parsers/ → rag/ingestion/ → rag/ → project_root
            project_root = Path(__file__).parent.parent.parent.parent
            path = project_root / whisper_model

        if not path.exists():
            raise FileNotFoundError(
                f"Whisper model not found: {path}\n"
                f"Run `motif setup` to download it.\n"
                f"(Configured whisper path: {whisper_model!r})"
            )
        return path

    def _transcribe(self, audio_path: Path, model_path: Path) -> List[dict]:
        """
        Run whisper.cpp transcription. Returns list of segment dicts:
        {"text": str, "start": float, "end": float}
        """
        try:
            from pywhispercpp.model import Model  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "pywhispercpp is not installed. Run: pip install pywhispercpp"
            ) from exc
            
        log.info("Loading Whisper model from %s", model_path)
        with suppress_c_stderr():
            model = Model(str(model_path), n_threads=self._config.llm.threads, print_realtime=False, print_progress=False, print_timestamps=False)
            
            log.info("Transcribing %s", audio_path.name)
            segments = model.transcribe(str(audio_path))
        
        return [
            {
                "text": seg.text.strip(),
                "start": seg.t0 / 100.0,   # pywhispercpp times in centiseconds
                "end": seg.t1 / 100.0,
            }
            for seg in segments
            if seg.text.strip()
        ]

    def _group_segments(self, segments: List[dict]) -> List[ParsedPage]:
        """
        Group whisper segments into chunks of approximately TARGET_WORDS words.
        Each group becomes one ParsedPage with start_time/end_time set.
        """
        pages = []
        current_texts = []
        current_word_count = 0
        chunk_start_time = segments[0]["start"]

        for seg in segments:
            words = seg["text"].split()
            if current_word_count + len(words) > TARGET_WORDS and current_texts:
                pages.append(ParsedPage(
                    text=" ".join(current_texts),
                    start_time=chunk_start_time,
                    end_time=segments[segments.index(seg) - 1]["end"],
                    is_ocr=False,
                ))
                current_texts = [seg["text"]]
                current_word_count = len(words)
                chunk_start_time = seg["start"]
            else:
                current_texts.append(seg["text"])
                current_word_count += len(words)

        if current_texts:
            pages.append(ParsedPage(
                text=" ".join(current_texts),
                start_time=chunk_start_time,
                end_time=segments[-1]["end"],
                is_ocr=False,
            ))

        return pages
