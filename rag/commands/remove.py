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
            console.print("[structure]No documents are currently indexed.[/structure]")
            return
            
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Indexed Document", style="cyan")
        for doc in docs:
            table.add_row(doc["filepath"])
        
        console.print(table)
        console.print("\n[structure]Usage: /remove <substring of path>[/structure]")
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
            console.print(f"[warning]Ambiguous query '{query}'. Matches multiple documents:[/warning]")
            for m in matches:
                console.print(f"  - {m}")
            return
        else:
            console.print(f"[error]No indexed document matches '{query}'.[/error]")
            return

    try:
        from rag.ingestion import remove_document
        removed = remove_document(target_path, config=config)
        console.print(f"[success]Removed[/success] {removed} chunks for [structure]{target_path.name}[/structure].")
    except ImportError:
        console.print(
            f"[warning]Remove not yet implemented[/warning] (Phase 2).\n"
            f"Target: [structure]{target_path}[/structure]"
        )
