"""rag/commands/remove.py — /remove command."""
from __future__ import annotations

import os
from pathlib import Path


def handle_remove(args, session, config, console) -> None:
    """
    /remove PATH

    Remove a document and all its indexed data from the knowledge base.
    The document file itself is not deleted from disk.
    Supports substring matching if the exact path isn't specified.
    """
    from rich import box
    from rich.table import Table

    from rag.storage.ingestion_tracker import IngestionTracker

    if not args:
        tracker = IngestionTracker(config)
        docs = tracker.list_all()
        tracker.close()
        
        if not docs:
            console.print("[subtle]No documents are currently indexed.[/subtle]")
            return
            
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Indexed Document", style="bold")
        for doc in docs:
            table.add_row(doc["filepath"])
        
        console.print(table)
        console.print("\n[subtle]Usage: /remove <filename or path>[/subtle]")
        return

    query = args[0]
    target_path = None
    
    tracker = IngestionTracker(config)
    docs = tracker.list_all()
    tracker.close()
    
    exact_match = Path(os.path.expanduser(query)).resolve()
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
        remove_document(target_path, config=config)
        console.print(f"[success]Removed[/success] [bold]{target_path.name}[/bold] from the knowledge base.")
    except Exception as exc:
        from rag.errors import humanize_error
        console.print(f"[error]Could not remove document:[/error] {humanize_error(exc)}")
