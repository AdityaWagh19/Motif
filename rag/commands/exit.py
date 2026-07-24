"""rag/commands/exit.py — /exit and /quit commands."""
from __future__ import annotations

import sys


def handle_exit(args, session, config, console) -> None:
    """
    /exit or /quit — Save session and exit the application.
    """
    session.save()
    console.print("[subtle]Goodbye.[/subtle]")
    sys.exit(0)
