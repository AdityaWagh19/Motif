"""rag/commands/reset.py — /reset command handler."""
from __future__ import annotations

import shutil
from pathlib import Path


def handle_reset(args, session, config, console) -> None:
    """
    /reset — Purge all database indices, caches, and session history in the workspace.
    """
    from prompt_toolkit import prompt
    from prompt_toolkit.formatted_text import HTML

    try:
        confirm = prompt(
            HTML("<style fg='#FF2E93'>?</style> Are you sure you want to reset and purge all workspace data? [y/N]: ")
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return

    if confirm not in ("y", "yes"):
        console.print("[subtle]Reset cancelled.[/subtle]")
        return

    try:
        from rag.storage.db_manager import DatabaseManager
        DatabaseManager.close_all()

        ws_dir = config.db_root
        if ws_dir.exists():
            for item in ws_dir.iterdir():
                if item.name == "models":
                    continue  # Keep downloaded models
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception:
                    pass

        session.clear()
        session.flush_cache()
        console.print("[success]Workspace data and index purged cleanly.[/success]")
    except Exception as exc:
        from rag.errors import humanize_error
        console.print(f"[error]Could not reset workspace:[/error] {humanize_error(exc)}")
