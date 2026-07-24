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
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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

@dataclass
class ParentChunkerConfig:
    """7-B: Configuration for ParentChunker."""
    parent_tokens: int = 512
    child_tokens: int = 128
    overlap_tokens: int = 32


# ---------------------------------------------------------------------------
# SentenceChunker (now Semantic/Structural Chunker)
# ---------------------------------------------------------------------------

class SentenceChunker:
    """
    Splits text on structural markdown boundaries, falling back to recursive character splitting.

    Target: ~512 tokens per chunk.
    Overlap: ~64 tokens from the end of the previous chunk.

    Replaces the legacy naive sentence splitter.
    """

    def __init__(self, config: ChunkerConfig = ChunkerConfig()) -> None:
        self._target_chars: int = int(config.target_tokens * 4)  # rough char approx
        self._overlap_chars: int = int(config.overlap_tokens * 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        page: ParsedPage,
        source: str,
        filename: str,
        source_type: str,
    ) -> list[Chunk]:
        """
        Chunk a single ParsedPage into Chunk objects using RecursiveCharacterTextSplitter.
        """
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

        text = page.text.strip()
        if not text:
            return []

        # 1. Split by Markdown Headers if present
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        try:
            md_docs = markdown_splitter.split_text(text)
        except Exception:
            # Fallback if markdown splitter fails for some reason
            from langchain_core.documents import Document
            md_docs = [Document(page_content=text)]

        # 2. Sub-chunk using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._target_chars,
            chunk_overlap=self._overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks: list[Chunk] = []

        for doc in md_docs:
            sub_chunks = text_splitter.split_text(doc.page_content)
            for sub_text in sub_chunks:
                # Reconstruct metadata (Header path)
                header_path = " > ".join(v for k, v in doc.metadata.items() if k.startswith("Header"))
                section_val = header_path if header_path else page.section
                
                chunk_text = sub_text
                # Inject hierarchical metadata into text for better embedding
                if header_path:
                    chunk_text = f"Document: {filename}. Path: {header_path}.\n{sub_text}"

                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source=source,
                    filename=filename,
                    source_type=source_type,
                    page=page.page,
                    section=section_val,
                    has_table=page.has_table,
                    has_image=page.has_image,
                    is_ocr=page.is_ocr,
                    start_time=page.start_time,
                    end_time=page.end_time,
                    token_count=len(chunk_text.split()),  # word-count approx
                    indexed_at=datetime.now(UTC).isoformat(),
                ))

        return chunks

    def chunk_pages(
        self,
        pages: list[ParsedPage],
        source: str,
        filename: str,
        source_type: str,
    ) -> list[Chunk]:
        """
        Chunk all pages from a parsed document.
        """
        all_chunks: list[Chunk] = []
        # In a real global markdown splitter, we'd join all pages first, but 
        # to preserve page metadata (is_ocr, has_image, page_num), we chunk per page.
        # The Header path might reset per page if no header exists on that page,
        # but this is a solid structural improvement over naive regex.
        for page in pages:
            all_chunks.extend(self.chunk(page, source, filename, source_type))
        return all_chunks

# ---------------------------------------------------------------------------
# ParentChunker (7-B)
# ---------------------------------------------------------------------------

class ParentChunker:
    """
    7-B: Parent-Document Retrieval.
    Chunks text into 512-token parent chunks, then sub-chunks them into 
    128-token child chunks. The child chunks get a `parent_id` linking 
    them to their parent.
    
    Returns a combined list of Parent chunks and Child chunks.
    (Parent chunks will NOT be embedded by the ingestion pipeline, but 
    saved in ChunkStore).
    """
    def __init__(self, config: ParentChunkerConfig = ParentChunkerConfig()) -> None:
        self._parent_chunker = SentenceChunker(ChunkerConfig(
            target_tokens=config.parent_tokens,
            overlap_tokens=0  # Parents don't overlap to avoid duplication
        ))
        self._child_chunker = SentenceChunker(ChunkerConfig(
            target_tokens=config.child_tokens,
            overlap_tokens=config.overlap_tokens
        ))

    def chunk_pages(
        self,
        pages: list[ParsedPage],
        source: str,
        filename: str,
        source_type: str,
    ) -> list[Chunk]:
        
        # 1. Chunk into parents
        parent_chunks = self._parent_chunker.chunk_pages(pages, source, filename, source_type)
        all_chunks = list(parent_chunks)
        
        from rag.ingestion.parsers.base import ParsedPage
        
        # 2. For each parent, chunk it into children
        for parent in parent_chunks:
            # Wrap parent text into a mock ParsedPage to feed child chunker
            parent_page = ParsedPage(
                page=parent.page,
                section=parent.section,
                text=parent.text,
                has_table=parent.has_table,
                has_image=parent.has_image,
                is_ocr=parent.is_ocr,
                start_time=parent.start_time,
                end_time=parent.end_time
            )
            
            child_chunks = self._child_chunker.chunk(parent_page, source, filename, source_type)
            
            # Link children to parent
            for child in child_chunks:
                child.parent_id = parent.id
                all_chunks.append(child)
                
        return all_chunks
