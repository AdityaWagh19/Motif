"""
rag/ingestion/parsers/csv.py — pandas-based CSV parser.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rag.ingestion.parsers.base import BaseParser, ParsedPage

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)


class CSVParser(BaseParser):
    """
    Parser for CSV files. Converts rows into structured text for embeddings.
    """

    SUPPORTED_EXTENSIONS = [".csv", ".tsv"]

    def __init__(self, config: RAGConfig | None = None) -> None:
        self._config = config

    def parse(self, path: Path) -> list[ParsedPage]:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "pandas is not installed. Run: uv pip install pandas"
            ) from exc

        try:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(str(path), sep=sep, engine="python", on_bad_lines="skip")
        except Exception as exc:
            raise RuntimeError(f"Failed to parse CSV {path}: {exc}") from exc

        if df.empty:
            return []

        # Yield a ParsedPage every 50 rows to prevent massive chunks
        pages = []
        batch_size = 50
        for start_idx in range(0, len(df), batch_size):
            batch_df = df.iloc[start_idx:start_idx + batch_size]
            text_blocks = []
            for i, row in batch_df.iterrows():
                row_str = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                text_blocks.append(f"Row {i + 1}: {row_str}")
            
            text = "\n".join(text_blocks)
            pages.append(ParsedPage(text=text, has_table=True))
            
        return pages
