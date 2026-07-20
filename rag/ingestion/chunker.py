"""
rag/ingestion/chunker.py — SentenceChunker: fixed-size overlapping text chunks.

Splits ParsedPage objects into Chunk objects targeting ~512 tokens each with
~64 tokens of overlap between consecutive chunks. Token count is approximated
as word count (accurate enough for chunking, zero additional dependencies).

Phase 4 adds SemanticChunker using embedding cosine distance.

Dependency graph position:
    chunker  →  rag.types  (Chunk)
    chunker  →  rag.ingestion.parsers.base  (ParsedPage) — TYPE_CHECKING only
    chunker  →  (stdlib: re, uuid, datetime)
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List

from rag.types import Chunk

if TYPE_CHECKING:
    from rag.ingestion.parsers.base import ParsedPage


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Words-per-token approximation: English prose ≈ 0.75 tokens per word.
# We use word count as a cheap proxy — no tokenizer needed at chunk time.
_WORDS_PER_TOKEN: float = 0.75

# Sentence boundary: split AFTER . ! ? followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkerConfig:
    """Configuration for SentenceChunker."""
    target_tokens: int = 512   # Target chunk size in tokens
    overlap_tokens: int = 64   # Overlap between consecutive chunks in tokens


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------

class SentenceChunker:
    """
    Splits text on sentence boundaries into overlapping fixed-size chunks.

    Target: ~512 tokens per chunk (≈682 words).
    Overlap: ~64 tokens from the end of the previous chunk (≈85 words).

    The overlap ensures that a fact split across a chunk boundary can still
    be retrieved by either chunk.

    Usage:
        chunker = SentenceChunker()
        chunks = chunker.chunk_pages(pages, source="/path/to/doc.pdf",
                                     filename="doc.pdf", source_type="pdf")
    """

    def __init__(self, config: ChunkerConfig = ChunkerConfig()) -> None:
        self._target_words: int = max(1, int(config.target_tokens / _WORDS_PER_TOKEN))
        self._overlap_words: int = max(0, int(config.overlap_tokens / _WORDS_PER_TOKEN))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        page: "ParsedPage",
        source: str,
        filename: str,
        source_type: str,
    ) -> List[Chunk]:
        """
        Chunk a single ParsedPage into Chunk objects.

        Algorithm:
        1. Split page text into sentences on [.!?] followed by whitespace.
        2. Accumulate sentences until word count ≥ target_words.
        3. Emit a chunk; the next chunk starts with the last overlap_words
           words from the just-emitted chunk (sliding window overlap).
        4. Assign a fresh UUID to each chunk.
        5. Set token_count = word count (approximation).

        Args:
            page:        Parsed page/section to chunk.
            source:      Absolute file path string (stored in Chunk.source).
            filename:    Bare filename (e.g. "report.pdf").
            source_type: MIME-like type string ("pdf", "md", "txt", …).

        Returns:
            List of Chunk objects. Returns [] if page.text is empty.
        """
        text = page.text.strip()
        if not text:
            return []

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_word_count: int = 0

        def _emit_chunk(sents: List[str]) -> Chunk:
            chunk_text = " ".join(sents)
            return Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                source=source,
                filename=filename,
                source_type=source_type,
                page=page.page,
                section=page.section,
                has_table=page.has_table,
                has_image=page.has_image,
                is_ocr=page.is_ocr,
                start_time=page.start_time,
                end_time=page.end_time,
                token_count=len(chunk_text.split()),  # word-count approximation
                indexed_at=datetime.now(timezone.utc).isoformat(),
            )

        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)

            # If adding this sentence would exceed the target AND we already
            # have content — emit the current chunk first.
            if current_word_count + word_count > self._target_words and current_sentences:
                chunks.append(_emit_chunk(current_sentences))

                # Build overlap: take the last overlap_words words from the
                # chunk we just emitted, then start fresh with this sentence.
                all_prev_words = " ".join(current_sentences).split()
                overlap_words = all_prev_words[-self._overlap_words:] if self._overlap_words else []

                if overlap_words:
                    overlap_text = " ".join(overlap_words)
                    current_sentences = [overlap_text, sentence]
                    current_word_count = len(overlap_words) + word_count
                else:
                    current_sentences = [sentence]
                    current_word_count = word_count
            else:
                current_sentences.append(sentence)
                current_word_count += word_count

        # Flush the final partial chunk (always non-empty here).
        if current_sentences:
            chunks.append(_emit_chunk(current_sentences))

        return chunks

    def chunk_pages(
        self,
        pages: "List[ParsedPage]",
        source: str,
        filename: str,
        source_type: str,
    ) -> List[Chunk]:
        """
        Chunk all pages from a parsed document.

        Returns a flat list of Chunk objects in page order.
        """
        all_chunks: List[Chunk] = []
        for page in pages:
            all_chunks.extend(self.chunk(page, source, filename, source_type))
        return all_chunks
