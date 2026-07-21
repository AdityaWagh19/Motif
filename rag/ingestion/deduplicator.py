"""
rag/ingestion/deduplicator.py — SimHash near-duplicate chunk detection.

Prevents nearly-identical content (e.g. repeated boilerplate headers, footers,
or copyright notices) from bloating the index and inflating retrieval scores.

Algorithm:
  - Compute a 64-bit SimHash fingerprint from character trigrams of each chunk's text.
  - Two chunks are near-duplicates if their fingerprints differ by ≤ threshold bits
    (Hamming distance). Default threshold: 3 bits (≈95% similarity).

Scope:
  - In-memory only — the seen-hashes set is NOT persisted between ingestion runs.
  - Cross-run deduplication is handled by IngestionTracker (file-level SHA-256).
  - Reset between documents with reset() to avoid cross-document deduplication.

Dependency:
    simhash >= 2.0  (pip install simhash)

Dependency graph position:
    deduplicator  →  simhash  (third-party)
    deduplicator  →  rag.types  (Chunk)
"""
from __future__ import annotations

import logging

from rag.types import Chunk

log = logging.getLogger(__name__)

# Hamming distance threshold for near-duplicate detection.
# 3 bits out of 64 ≈ 95% text similarity required to be considered a duplicate.
DUPLICATE_HAMMING_THRESHOLD: int = 3


class Deduplicator:
    """
    SimHash-based near-duplicate chunk detector.

    Maintains an in-memory list of (Simhash, chunk_id) pairs seen during
    a single ingestion session. Keeps the first occurrence of each near-duplicate
    cluster; subsequent occurrences are filtered out.

    Usage:
        dedup = Deduplicator()
        for doc_pages in documents:
            chunks = chunker.chunk_pages(doc_pages, ...)
            unique = dedup.filter(chunks)
            dedup.reset()   # reset between documents
    """

    def __init__(self, threshold: int = DUPLICATE_HAMMING_THRESHOLD) -> None:
        self._threshold = threshold
        self._seen: list[tuple[object, str]] = []   # (Simhash, chunk_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_hash(self, text: str) -> object:
        """
        Compute a SimHash fingerprint from character trigrams of *text*.

        Character trigrams capture local similarity better than word n-grams
        for short text chunks and are robust to minor wording changes.
        """
        try:
            from simhash import Simhash  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "simhash is not installed. Run: pip install simhash"
            ) from exc

        # Generate character trigrams: "abc" → ["abc", "bc_", ...] (no padding needed)
        trigrams = [text[i:i + 3] for i in range(len(text) - 2)] if len(text) >= 3 else [text]
        return Simhash(trigrams)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(self, chunk: Chunk) -> bool:
        """
        Return True if *chunk* is a near-duplicate of any previously seen chunk.

        Side-effect: if NOT a duplicate, adds this chunk's fingerprint to the
        seen set so that future near-duplicates of it will be detected.

        Args:
            chunk: The chunk to test.

        Returns:
            True  → chunk is a near-duplicate (caller should skip it).
            False → chunk is unique; its fingerprint has been recorded.
        """
        h = self._compute_hash(chunk.text)
        for seen_hash, seen_id in self._seen:
            if h.distance(seen_hash) <= self._threshold:  # type: ignore[union-attr]
                log.debug(
                    "Deduplicator: chunk %s is near-duplicate of %s — skipping.",
                    chunk.id,
                    seen_id,
                )
                return True
        self._seen.append((h, chunk.id))
        return False

    def filter(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Return a new list with near-duplicate chunks removed.

        The first occurrence of each near-duplicate cluster is kept;
        subsequent occurrences are dropped.

        Args:
            chunks: List of chunks to filter.

        Returns:
            Filtered list (may be shorter than input).
        """
        result: list[Chunk] = []
        for chunk in chunks:
            if not self.is_duplicate(chunk):
                result.append(chunk)
        dropped = len(chunks) - len(result)
        if dropped > 0:
            log.debug("Deduplicator: dropped %d near-duplicate chunk(s).", dropped)
        return result

    def reset(self) -> None:
        """
        Clear the seen-fingerprints set.

        Call this between documents to prevent cross-document deduplication
        (two documents may legitimately share similar boilerplate).
        """
        self._seen = []
