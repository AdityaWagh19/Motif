"""
rag/session.py — Conversation session management.

Manages the in-session conversation history (a rolling list of Q&A turns),
persists it to ~/.ragdb/history.json on exit, and loads it on restart.

Design decisions (see context.md):
- Single-user, local, offline — no session IDs, no database, no server.
- History is a plain Python list of {"role": str, "content": str} dicts.
- Retrieved document passages always take priority over history in the
  context budget. Oldest turns are dropped first when over budget.
- history.json is the only persistent state; the document index is in Qdrant/SQLite.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig

# Default storage location (mirrors StorageConfig.db_path default)
_DEFAULT_DB_ROOT = Path("~/.ragdb").expanduser()
_HISTORY_FILENAME = "history.json"


class Session:
    """
    Lightweight session: conversation history + config reference.

    Attributes:
        history:  List of {"role": "user"|"assistant", "content": str} dicts.
                  Ordered oldest-first. Append via add_turn().
        config:   Loaded RAGConfig (set by caller after load_config()).
        db_root:  Path to ~/.ragdb (or override from config).
    """

    def __init__(self, config: "RAGConfig | None" = None) -> None:
        self.config = config
        self.history: list[dict[str, str]] = []
        self._db_root: Path = (
            config.db_root if config is not None else _DEFAULT_DB_ROOT
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def history_path(self) -> Path:
        return self._db_root / _HISTORY_FILENAME

    @property
    def last_query(self) -> str | None:
        """Return the most recent user query, or None if history is empty."""
        for turn in reversed(self.history):
            if turn["role"] == "user":
                return turn["content"]
        return None

    @property
    def turn_count(self) -> int:
        """Number of complete Q&A exchanges (each exchange = 2 entries)."""
        return sum(1 for t in self.history if t["role"] == "user")

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_turn(self, query: str, answer: str) -> None:
        """
        Append a completed Q&A exchange to the history.

        Args:
            query:  The user's raw query string.
            answer: The LLM-generated answer string (citations stripped).
        """
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})

    # ── Context budget management ─────────────────────────────────────────────

    def get_history_for_context(
        self,
        token_budget: int,
        passage_tokens: int,
        chars_per_token: int = 4,
    ) -> list[dict[str, str]]:
        """
        Return a rolling window of history turns that fits within the token budget
        AFTER the retrieved passages have been allocated their space.

        Oldest turns are dropped first. Retrieved passages always take priority.

        Args:
            token_budget:    Total context window token budget (e.g. 2048).
            passage_tokens:  Tokens already consumed by retrieved passages.
            chars_per_token: Approximate characters per token (default: 4).

        Returns:
            A list of {"role", "content"} dicts (oldest-first), truncated to fit.
        """
        remaining = token_budget - passage_tokens
        if remaining <= 0 or not self.history:
            return []

        # Walk history newest-first, accumulating until budget exhausted
        selected: list[dict[str, str]] = []
        for turn in reversed(self.history):
            turn_tokens = len(turn["content"]) // chars_per_token
            if turn_tokens > remaining:
                break
            selected.append(turn)
            remaining -= turn_tokens

        # Return in chronological order (oldest-first)
        return list(reversed(selected))

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Write the current history to ~/.ragdb/history.json.
        Creates the db_root directory if it does not exist.
        Safe to call on an empty history (writes an empty list).
        """
        self._db_root.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """
        Load history from ~/.ragdb/history.json if it exists.

        Returns:
            True if history was loaded successfully, False if the file
            did not exist or could not be parsed.
        """
        if not self.history_path.exists():
            return False
        try:
            with open(self.history_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.history = data
                return True
        except (json.JSONDecodeError, OSError):
            pass
        return False

    def clear(self) -> None:
        """
        Clear in-memory history and delete ~/.ragdb/history.json.
        Implements the /clear slash command.
        """
        self.history = []
        if self.history_path.exists():
            self.history_path.unlink()

    def new(self) -> Path | None:
        """
        Archive the current history to history_YYYYMMDD_HHMMSS.json,
        then start a fresh empty session. Implements the /new slash command.

        Returns:
            Path to the archived file, or None if there was no history to archive.
        """
        if not self.history:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self._db_root / f"history_{timestamp}.json"
        if self.history_path.exists():
            shutil.copy2(self.history_path, archive_path)

        self.clear()
        return archive_path
