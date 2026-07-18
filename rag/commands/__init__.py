"""
rag/commands/__init__.py — Slash command registry.

All slash commands are registered here. The REPL routes /command strings
through get_command(), which returns the appropriate handler callable.

Each handler has the signature:
    handler(args: list[str], session: Session, config: RAGConfig, console: Console) -> None
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console
    from rag.session import Session
    from rag.config import RAGConfig

# Import all command handlers
from rag.commands.help import handle_help
from rag.commands.clear import handle_clear, handle_new
from rag.commands.status import handle_status
from rag.commands.ingest import handle_ingest
from rag.commands.remove import handle_remove
from rag.commands.sync import handle_sync
from rag.commands.setup import handle_setup

# ── Command registry ──────────────────────────────────────────────────────────
# Maps slash command strings to (handler, one-line description)
SLASH_COMMANDS: dict[str, tuple[Callable, str]] = {
    "/ingest":  (handle_ingest,  "Add documents to the knowledge base"),
    "/remove":  (handle_remove,  "Remove a document and all its chunks"),
    "/sync":    (handle_sync,    "Sync a directory: add new, remove deleted, re-index changed"),
    "/status":  (handle_status,  "Show index statistics and loaded model info"),
    "/clear":   (handle_clear,   "Clear conversation history and delete history.json"),
    "/new":     (handle_new,     "Archive current history and start a fresh session"),
    "/setup":   (handle_setup,   "Download models for your hardware tier"),
    "/help":    (handle_help,    "Show all available commands"),
}

COMMAND_DESCRIPTIONS: dict[str, str] = {k: v[1] for k, v in SLASH_COMMANDS.items()}


def get_command(name: str) -> Callable | None:
    """Return the handler for a slash command, or None if not registered."""
    entry = SLASH_COMMANDS.get(name.lower())
    return entry[0] if entry else None
