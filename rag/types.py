"""
rag/types.py — Shared data contracts for all cross-module data.

RULE: Every function that returns data crossing a module boundary must use a
type defined here. No module defines its own result types externally.

Import order: rag.types has no internal rag dependencies — stdlib only.
This guarantees it can be imported by every other module without circular deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Storage / Ingestion types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single indexed unit of text.

    Stored in:
      - ChunkStore (SQLite) — text + metadata
      - Qdrant — vector embedding + payload (subset of fields for filtering)
      - BM25Index — text only
    """
    id: str                              # UUID (primary key)
    text: str                            # Raw chunk text as extracted and cleaned
    source: str                          # Absolute filepath to source document
    filename: str                        # Basename of source file
    source_type: str                     # "pdf" | "docx" | "md" | "image" | "audio"

    # Position within document
    char_start: int = 0
    char_end: int = 0
    page: Optional[int] = None           # PDF / DOCX page number (1-indexed)
    section: Optional[str] = None        # Nearest detected section heading

    # Audio-specific position
    start_time: Optional[float] = None   # seconds
    end_time: Optional[float] = None     # seconds

    # Content flags (used for routing and Qdrant payload filters)
    has_table: bool = False
    has_image: bool = False
    is_ocr: bool = False
    language: Optional[str] = None

    # Ingestion metadata
    content_hash: str = ""               # SHA-256 of text, for deduplication
    token_count: int = 0                 # Approximate token count (for context budgeting)
    indexed_at: str = ""                 # ISO 8601 UTC timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredPassage:
    """
    A retrieved Chunk with its associated relevance score.

    Lifecycle:
      1. After dense/sparse/BM25 retrieval: score = RRF score
      2. After reranking: score = cross-encoder score, method = "reranked"
    """
    chunk: Chunk
    score: float                         # Higher = more relevant
    retrieval_method: str                # "dense" | "sparse" | "bm25" | "reranked"


# ─────────────────────────────────────────────────────────────────────────────
# Generation types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """
    A source reference included in the LLM answer. Rendered as [N] inline.
    """
    number: int                          # Citation index (1-based)
    source_type: str                     # "pdf" | "docx" | "md" | "image" | "audio"
    filepath: str
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    start_time: Optional[float] = None   # Audio: seconds
    end_time: Optional[float] = None
    relevance_score: float = 0.0
    excerpt: str = ""                    # First ~150 characters of chunk text

    def format(self) -> str:
        """Render the citation as a single-line string."""
        base = f"[{self.number}] {self.filename}"
        if self.source_type == "audio" and self.start_time is not None:
            s = f"{int(self.start_time // 60):02d}:{int(self.start_time % 60):02d}"
            e = f"{int(self.end_time // 60):02d}:{int(self.end_time % 60):02d}"
            return f"{base} @ {s}–{e}"
        if self.source_type in ("pdf", "docx"):
            if self.page:
                base += f" (p.{self.page})"
            if self.section:
                base += f" — {self.section}"
        return base


@dataclass
class AnswerResult:
    """
    Returned by QueryPipeline.answer() and consumed directly by the REPL.
    The REPL streams `text` token-by-token; citations are printed after streaming ends.
    """
    text: str
    citations: list[Citation]
    passages_used: int
    used_hyde: bool = False
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tier: str = "T1"


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion operation results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """
    Returned by rag.ingestion.ingest_path() and consumed by /ingest command.
    """
    files_processed: int
    chunks_added: int
    files_skipped: int                   # Already indexed (dedup / hash unchanged)
    errors: list[str] = field(default_factory=list)


@dataclass
class SyncResult:
    """
    Returned by rag.ingestion.sync_directory() and consumed by /sync command.
    """
    added: int
    removed: int
    reindexed: int
    errors: list[str] = field(default_factory=list)
