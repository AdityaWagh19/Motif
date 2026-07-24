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
            console.print("[subtle]No workspaces found.[/subtle]")
            return
            
        workspaces = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "models"])
        
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Workspace", style="bold")
        table.add_column("Active", style="success", justify="center")
        
        for w in workspaces:
            is_active = (w == config.storage.workspace)
            name_str = f"[accent_bold]{w}[/accent_bold]" if is_active else w
            active_str = "✓" if is_active else ""
            table.add_row(name_str, active_str)
            
        console.print()
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

        doc_count = 0
        try:
            from rag.storage.chunk_store import ChunkStore
            with ChunkStore(config) as store:
                doc_count = store.count_documents()
        except Exception:
            pass

        doc_label = f"{doc_count} document{'s' if doc_count != 1 else ''}"
        console.print(f"[success]Switched to '{name}'[/success]  ·  [dim]{doc_label}[/dim]")
        
    elif subcmd == "delete":
        if len(args) < 2:
            console.print("[error]Usage:[/error] /workspace delete <name>")
            return
        name = args[1]
        
        if name == config.storage.workspace:
            console.print("[error]Cannot delete active workspace.[/error] Switch to another workspace first.")
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
            confirm = prompt(HTML(f"<style fg='#FF2E93'>?</style> Are you sure you want to delete workspace '<b>{name}</b>'? [y/N]: ")).strip().lower()
        except (KeyboardInterrupt, EOFError):
            return

        if confirm not in ("y", "yes"):
            console.print("[subtle]Deletion cancelled.[/subtle]")
            return

        try:
            shutil.rmtree(target)
            console.print(f"[success]Deleted workspace '{name}'.[/success]")
        except Exception as exc:
            from rag.errors import humanize_error
            console.print(f"[error]Could not delete workspace '{name}':[/error] {humanize_error(exc)}")
        
    else:
        console.print(f"[error]Unknown subcommand: {subcmd}[/error]")
        console.print("[error]Usage:[/error] /workspace list | new <name> | switch <name> | delete <name>")
