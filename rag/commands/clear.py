"""rag/commands/clear.py — /clear and /new commands."""
from __future__ import annotations


def handle_clear(args, session, config, console) -> None:
    """
    /clear — Wipe in-memory conversation history and delete history.json.

    The document index is unaffected. Only the conversation history is cleared.
    """
    count = session.turn_count
    session.clear()
    console.print(
        f"[success]Cleared[/success] {count} exchange{'s' if count != 1 else ''} "
        f"from conversation history."
    )


def handle_new(args, session, config, console) -> None:
    """
    /new — Archive the current history and start a fresh session.

    The current history is saved to history_YYYYMMDD_HHMMSS.json before clearing.
    Also flushes the retrieval cache.
    """
    archive_path = session.new()
    session.flush_cache()
    if archive_path:
        console.print(
            f"[success]History archived[/success] to [structure]{archive_path.resolve()}[/structure].\n"
            "Query cache flushed. Starting fresh session."
        )
    else:
        console.print("[structure]No history to archive. Query cache flushed. Starting fresh session.[/structure]")
