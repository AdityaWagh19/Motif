"""rag/commands/sync.py — /sync command. (Phase 2 implementation pending)"""
from __future__ import annotations

import argparse
from pathlib import Path


def handle_sync(args, session, config, console) -> None:
    """
    /sync DIR [-r]

    Sync a directory with the knowledge base:
      - New files are ingested.
      - Deleted files are removed from the index.
      - Changed files (content hash differs) are re-indexed.

    Phase 0: stub — wired to rag.ingestion.sync_directory() in Phase 2.
    """
    parser = argparse.ArgumentParser(prog="/sync", add_help=False)
    parser.add_argument("directory", nargs="?")
    parser.add_argument("-r", "--recursive", action="store_true")

    try:
        parsed = parser.parse_args(args)
    except SystemExit:
        console.print("[error]Usage:[/error] /sync DIR [-r]")
        return

    if not parsed.directory:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import PathCompleter
        from prompt_toolkit.formatted_text import HTML
        
        try:
            path_str = prompt(
                HTML("<ansiblue>?</ansiblue> Enter directory to sync: "),
                completer=PathCompleter(expanduser=True, only_directories=True)
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return
            
        if not path_str:
            return
        parsed.directory = path_str

    target = Path(str(parsed.directory)).expanduser().resolve()  # type: ignore[attr-defined]
    if not target.is_dir():
        console.print(f"[error]Not a directory:[/error] {target}")
        return

    try:
        from rag.ingestion import sync_directory
        result = sync_directory(target, config=config, recursive=parsed.recursive, console=console)
        console.print(
            f"[success]Sync complete.[/success] "
            f"Added: {result.added}  Removed: {result.removed}  Re-indexed: {result.reindexed}"
        )
    except ImportError:
        console.print(
            f"[warning]Sync not yet implemented[/warning] (Phase 2).\n"
            f"Target: [structure]{target}[/structure]  "
            f"Recursive: {'yes' if parsed.recursive else 'no'}"
        )
