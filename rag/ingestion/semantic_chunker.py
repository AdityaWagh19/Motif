"""
rag/ingestion/semantic_chunker.py — Embedding-based semantic chunking.

Available on T2 and T3 only (config.chunking.use_semantic = True).
Falls back to SentenceChunker on T1 (time budget is too tight for 
double-embedding every document).

Algorithm (cosine-distance segmentation):
  1. Sentence-split the page text.
  2. Embed each sentence using the Embedder.
  3. Compute cosine distance between consecutive sentence embeddings.
  4. Start a new chunk whenever distance > threshold (default 0.3).
  5. If a resulting segment is larger than target_tokens, further split it
     using the SentenceChunker as a fallback.

Why not use the semantic-text-splitter Rust crate?
  - It requires a compiled tokenizer that varies per model.
  - Our approach (embed-then-split) is model-agnostic and already has the
    embedder loaded in memory during ingestion.

Reference: Greg Kamradt's "5 Levels of Text Splitting" (2023).
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from rag.types import Chunk
from rag.ingestion.chunker import SentenceChunker, ChunkerConfig, _SENTENCE_SPLIT_RE

if TYPE_CHECKING:
    from rag.ingestion.parsers.base import ParsedPage
    from rag.models.embedder import Embedder
    from rag.config import RAGConfig


# Minimum number of sentences per semantic segment before it is emitted as a
# chunk. This avoids tiny one-sentence chunks when text is very diverse.
_MIN_SENTENCES_PER_CHUNK = 2

# Maximum words in a semantic segment before it is forcibly sub-split.
_WORDS_PER_TOKEN: float = 0.75


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance (1 - cosine_similarity) between two L2-normed vectors.

    Args:
        a: L2-normalised float32 vector.
        b: L2-normalised float32 vector.

    Returns:
        float in [0.0, 2.0]. Lower = more similar.
    """
    # Both vectors are already L2-normalised by the Embedder.
    dot = float(np.dot(a, b))
    # Clamp to [-1, 1] to handle floating-point noise.
    dot = max(-1.0, min(1.0, dot))
    return 1.0 - dot


class SemanticChunker:
    """
    Splits text into topically coherent chunks using embedding distance.

    Usage:
        chunker = SemanticChunker(config, embedder)
        chunks = chunker.chunk_pages(pages, source=..., filename=..., source_type=...)
    """

    def __init__(self, config: "RAGConfig", embedder: "Embedder") -> None:
        """
        Initialise the semantic chunker.

        Args:
            config:   RAGConfig — reads chunking.semantic_threshold and
                      chunking.target_tokens.
            embedder: Loaded Embedder instance (already warmed up).
        """
        self._threshold = config.chunking.semantic_threshold
        self._target_words = max(1, int(config.chunking.target_tokens / _WORDS_PER_TOKEN))
        self._embedder = embedder
        # Fallback splitter for oversized segments.
        self._fallback = SentenceChunker(
            ChunkerConfig(
                target_tokens=config.chunking.target_tokens,
                overlap_tokens=config.chunking.overlap_tokens,
            )
        )

    def chunk_pages(
        self,
        pages: "List[ParsedPage]",
        source: str,
        filename: str,
        source_type: str,
    ) -> List[Chunk]:
        """
        Chunk all pages into semantically coherent chunks.

        Returns a flat list of Chunk objects in page order.
        """
        all_chunks: List[Chunk] = []
        for page in pages:
            all_chunks.extend(self._chunk_page(page, source, filename, source_type))
        return all_chunks

    def _chunk_page(
        self,
        page: "ParsedPage",
        source: str,
        filename: str,
        source_type: str,
    ) -> List[Chunk]:
        """
        Process a single ParsedPage into semantically coherent chunks.
        """
        text = page.text.strip()
        if not text:
            return []

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return []

        # Fall back to sentence chunker for very short pages.
        if len(sentences) < _MIN_SENTENCES_PER_CHUNK:
            return self._fallback.chunk(page, source, filename, source_type)

        # Embed each sentence for distance computation.
        # Use search_document: prefix (same as ingestion embedding).
        embeddings: np.ndarray = self._embedder.encode_batch(
            sentences,
            prefix="search_document: ",
        )

        # Segment sentences by cosine distance threshold.
        segments: List[List[str]] = []
        current: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            dist = _cosine_distance(embeddings[i - 1], embeddings[i])
            if dist > self._threshold and len(current) >= _MIN_SENTENCES_PER_CHUNK:
                segments.append(current)
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            segments.append(current)

        # Convert segments to Chunk objects.
        chunks: List[Chunk] = []
        for seg_sentences in segments:
            seg_text = " ".join(seg_sentences)
            word_count = len(seg_text.split())

            if word_count > self._target_words * 1.5:
                # Segment is too large — use SentenceChunker to sub-split it.
                from rag.ingestion.parsers.base import ParsedPage as PP
                sub_page = PP(
                    text=seg_text,
                    page=page.page,
                    section=page.section,
                    has_table=page.has_table,
                    has_image=page.has_image,
                    is_ocr=page.is_ocr,
                )
                sub_chunks = self._fallback.chunk(sub_page, source, filename, source_type)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=seg_text,
                    source=source,
                    filename=filename,
                    source_type=source_type,
                    page=page.page,
                    section=page.section,
                    has_table=page.has_table,
                    has_image=page.has_image,
                    is_ocr=page.is_ocr,
                    token_count=word_count,
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                ))

        return chunks
