"""
rag/ingestion/parsers/audio.py — Parser for audio (whisper.cpp transcription).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rag.ingestion.parsers.base import BaseParser, ParsedPage

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

class AudioParser(BaseParser):
    """
    Parser for audio files using whisperx.
    Groups transcripts by speaker turns (diarization).
    """
    
    SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

    def __init__(self, config: RAGConfig) -> None:
        self._config = config

    def parse(self, path: Path) -> list[ParsedPage]:
        """
        Transcribe audio file using whisperx.

        Returns one ParsedPage per speaker turn.
        Each ParsedPage carries start_time and end_time.
        """
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        segments = self._transcribe(path)

        if not segments:
            return []

        # Group segments by speaker turn
        return self._group_segments(segments)

    def _transcribe(self, audio_path: Path) -> list[dict]:
        """
        Run whisperx transcription, alignment, and diarization.
        """
        from rag.models.model_manager import get_model_manager
        import whisperx
        
        whisper_data = get_model_manager().get_whisper(self._config)
        model = whisper_data["model"]
        device = whisper_data["device"]
            
        log.info("Transcribing %s", audio_path.name)
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=16)
        
        # Align
        try:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        except Exception as e:
            log.warning("WhisperX alignment failed: %s", e)
            
        # Diarize
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                log.warning("WhisperX diarization failed: %s", e)
        else:
            log.info("Skipping WhisperX diarization (HF_TOKEN not found in environment)")

        return result.get("segments", [])

    def _group_segments(self, segments: list[dict]) -> list[ParsedPage]:
        """
        Group whisper segments into chunks by speaker turn.
        If diarization is missing or a single speaker speaks for a very long time,
        chunks will also be bounded by a ~350 word limit.
        """
        TARGET_WORDS = 350
        pages = []
        if not segments:
            return pages

        current_speaker = segments[0].get("speaker", "UNKNOWN")
        current_texts = []
        current_word_count = 0
        chunk_start_time = segments[0]["start"]

        for i, seg in enumerate(segments):
            speaker = seg.get("speaker", "UNKNOWN")
            words = seg["text"].split()
            
            if (speaker != current_speaker or current_word_count + len(words) > TARGET_WORDS) and current_texts:
                # Speaker shift or length limit -> emit chunk
                text = " ".join(current_texts)
                if current_speaker != "UNKNOWN":
                    text = f"[Speaker {current_speaker}]: {text}"
                pages.append(ParsedPage(
                    text=text,
                    start_time=chunk_start_time,
                    end_time=segments[i - 1]["end"],
                    is_ocr=False,
                ))
                current_texts = [seg["text"]]
                current_word_count = len(words)
                current_speaker = speaker
                chunk_start_time = seg["start"]
            else:
                current_texts.append(seg["text"])
                current_word_count += len(words)
                
        if current_texts:
            text = " ".join(current_texts)
            if current_speaker != "UNKNOWN":
                text = f"[Speaker {current_speaker}]: {text}"
            pages.append(ParsedPage(
                text=text,
                start_time=chunk_start_time,
                end_time=segments[-1]["end"],
                is_ocr=False,
            ))

        return pages
