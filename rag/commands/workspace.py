"""rag/commands/workspace.py — /workspace command."""
from __future__ import annotations

import shutil

from rich import box
from rich.table import Table


def handle_workspace(args, session, config, console) -> None:
    """
    /workspace — Manage isolated workspaces.
    
    Usage:
        /workspace list
        /workspace new <name>
        /workspace switch <name>
        /workspace delete <name>
    """
    if not args:
        console.print("[error]Usage:[/error] /workspace list | new <name> | switch <name> | delete <name>")
        return

    subcmd = args[0].lower()
    base_dir = config.db_root.parent

    if subcmd == "list":
        if not base_dir.exists():
            console.print("No workspaces found.")
            return
            
        workspaces = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "models"])
        
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Workspace", style="cyan")
        table.add_column("Active", style="green")
        
        for w in workspaces:
            active = "*" if w == config.storage.workspace else ""
            table.add_row(w, active)
            
        console.print(table)
        
    elif subcmd == "new":
        if len(args) < 2:
            console.print("[error]Usage:[/error] /workspace new <name>")
            return
        name = args[1]
        
        new_path = base_dir / name
        if new_path.exists():
            console.print(f"[error]Workspace '{name}' already exists.[/error]")
            return
            
        from rag.storage.db_manager import DatabaseManager
        DatabaseManager.close_all()
        new_path.mkdir(parents=True)
        config.storage.workspace = name
        config.save()
        
        session.clear()
        session.load()
        session.flush_cache()
        console.print(f"[success]Created and switched to workspace '{name}'.[/success]")
        
    elif subcmd == "switch":
        if len(args) < 2:
            console.print("[error]Usage:[/error] /workspace switch <name>")
            return
        name = args[1]
        
        new_path = base_dir / name
        if not new_path.exists():
            console.print(f"[error]Workspace '{name}' does not exist.[/error]")
            return
            
        from rag.storage.db_manager import DatabaseManager
        DatabaseManager.close_all()
        config.storage.workspace = name
        config.save()
        
        session.clear()
        session.load()
        session.flush_cache()

        chunk_count, doc_count = 0, 0
        try:
            from rag.storage.chunk_store import ChunkStore
            with ChunkStore(config) as store:
                chunk_count = store.count()
                doc_count = store.count_documents()
        except Exception:
            pass

        console.print(f"[success]✓ Switched to workspace '{name}' ({doc_count} docs, {chunk_count:,} chunks).[/success]")
        
    elif subcmd == "delete":
        if len(args) < 2:
            console.print("[error]Usage:[/error] /workspace delete <name>")
            return
        name = args[1]
        
        if name == config.storage.workspace:
            console.print("[error]Cannot delete active workspace.[/error] Switch first.")
            return
            
        if name == "default":
            console.print("[error]Cannot delete the 'default' workspace.[/error]")
            return
            
        target = base_dir / name
        if not target.exists():
            console.print(f"[error]Workspace '{name}' does not exist.[/error]")
            return
            
        from prompt_toolkit import prompt
        from prompt_toolkit.formatted_text import HTML
        try:
            confirm = prompt(HTML(f"<ansired>?</ansired> Are you sure you want to delete workspace '<b>{name}</b>' and all its data? [y/N]: ")).strip().lower()
        except (KeyboardInterrupt, EOFError):
            return

        if confirm not in ("y", "yes"):
            console.print("[structure]Deletion cancelled.[/structure]")
            return

        try:
            shutil.rmtree(target)
            console.print(f"[success]Deleted workspace '{name}'.[/success]")
        except Exception as exc:
            console.print(f"[error]Failed to delete workspace '{name}':[/error] {exc}")
        
    else:
        console.print(f"[error]Unknown subcommand: {subcmd}[/error]")
        console.print("[error]Usage:[/error] /workspace list | new <name> | switch <name> | delete <name>")
