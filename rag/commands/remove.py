"""rag/commands/remove.py — /remove command. (Phase 2 implementation pending)"""
from __future__ import annotations

from pathlib import Path


def handle_remove(args, session, config, console) -> None:
    """
    /remove PATH

    Remove a document and all its indexed chunks from the knowledge base.
    The document file itself is not deleted from disk.
    Supports substring matching if the exact path isn't found.
    """
    from rag.storage.ingestion_tracker import IngestionTracker
    from rich.table import Table
    from rich import box

    if not args:
        tracker = IngestionTracker(config)
        docs = tracker.list_all()
        tracker.close()
        
        if not docs:
            console.print("[dim]No documents are currently indexed.[/dim]")
            return
            
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Indexed Document", style="cyan")
        for doc in docs:
            table.add_row(doc["filepath"])
        
        console.print(table)
        console.print("\n[dim]Usage: /remove <substring of path>[/dim]")
        return

    query = args[0]
    target_path = None
    
    tracker = IngestionTracker(config)
    docs = tracker.list_all()
    tracker.close()
    
    exact_match = Path(query).expanduser().resolve()
    if any(d["filepath"] == str(exact_match) for d in docs):
        target_path = exact_match
    else:
        matches = [d["filepath"] for d in docs if query.lower() in d["filepath"].lower()]
        if len(matches) == 1:
            target_path = Path(matches[0])
        elif len(matches) > 1:
            console.print(f"[yellow]Ambiguous query '{query}'. Matches multiple documents:[/yellow]")
            for m in matches:
                console.print(f"  - {m}")
            return
        else:
            console.print(f"[red]No indexed document matches '{query}'.[/red]")
            return

    try:
        from rag.ingestion import remove_document
        removed = remove_document(target_path, config=config)
        console.print(f"[green]Removed[/green] {removed} chunks for [dim]{target_path.name}[/dim].")
    except ImportError:
        console.print(
            f"[yellow]Remove not yet implemented[/yellow] (Phase 2).\n"
            f"Target: [dim]{target_path}[/dim]"
        )
