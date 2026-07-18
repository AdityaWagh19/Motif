"""rag/commands/remove.py — /remove command. (Phase 2 implementation pending)"""
from __future__ import annotations

from pathlib import Path


def handle_remove(args, session, config, console) -> None:
    """
    /remove PATH

    Remove a document and all its indexed chunks from the knowledge base.
    The document file itself is not deleted from disk.

    Phase 0: stub — wired to rag.ingestion.remove_document() in Phase 2.
    """
    if not args:
        console.print("[red]Usage:[/red] /remove PATH")
        return

    target = Path(args[0]).expanduser().resolve()

    try:
        from rag.ingestion import remove_document
        removed = remove_document(target, config=config)
        console.print(f"[green]Removed[/green] {removed} chunks for [dim]{target.name}[/dim].")
    except ImportError:
        console.print(
            f"[yellow]Remove not yet implemented[/yellow] (Phase 2).\n"
            f"Target: [dim]{target}[/dim]"
        )
