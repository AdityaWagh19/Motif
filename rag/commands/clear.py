"""rag/commands/clear.py — /clear and /new commands."""
from __future__ import annotations


def handle_clear(args, session, config, console) -> None:
    """
    /clear — Wipe in-memory conversation history and delete history.json.

    The document index is unaffected. Only the conversation history is cleared.
    """
    count = session.turn_count
    if count == 0:
        console.print("[subtle]Nothing to clear.[/subtle]")
        return

    session.clear()
    console.print("[success]Conversation cleared.[/success]")


def handle_new(args, session, config, console) -> None:
    """
    /new — Archive current history and start a fresh session.
    """
    session.new()
    session.flush_cache()
    console.print("[success]Started a fresh conversation.[/success]")
