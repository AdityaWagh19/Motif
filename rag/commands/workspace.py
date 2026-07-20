"""rag/commands/workspace.py — /workspace command."""
from __future__ import annotations

import shutil
from pathlib import Path

from rich.table import Table
from rich import box

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
        console.print("[red]Usage:[/red] /workspace list | new <name> | switch <name> | delete <name>")
        return

    subcmd = args[0].lower()
    base_dir = Path(config.storage.db_path).expanduser().resolve()

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
            console.print("[red]Usage:[/red] /workspace new <name>")
            return
        name = args[1]
        
        new_path = base_dir / name
        if new_path.exists():
            console.print(f"[red]Workspace '{name}' already exists.[/red]")
            return
            
        new_path.mkdir(parents=True)
        config.storage.workspace = name
        config.save()
        
        # Flush session state for safety
        session.flush_cache()
        console.print(f"[green]Created and switched to workspace '{name}'.[/green]")
        
    elif subcmd == "switch":
        if len(args) < 2:
            console.print("[red]Usage:[/red] /workspace switch <name>")
            return
        name = args[1]
        
        new_path = base_dir / name
        if not new_path.exists():
            console.print(f"[red]Workspace '{name}' does not exist.[/red]")
            return
            
        config.storage.workspace = name
        config.save()
        
        # Flush session state
        session.flush_cache()
        console.print(f"[green]Switched to workspace '{name}'.[/green]")
        
    elif subcmd == "delete":
        if len(args) < 2:
            console.print("[red]Usage:[/red] /workspace delete <name>")
            return
        name = args[1]
        
        if name == config.storage.workspace:
            console.print("[red]Cannot delete active workspace.[/red] Switch first.")
            return
            
        if name == "default":
            console.print("[red]Cannot delete the 'default' workspace.[/red]")
            return
            
        target = base_dir / name
        if not target.exists():
            console.print(f"[red]Workspace '{name}' does not exist.[/red]")
            return
            
        shutil.rmtree(target, ignore_errors=True)
        console.print(f"[green]Deleted workspace '{name}'.[/green]")
        
    else:
        console.print(f"[red]Unknown subcommand: {subcmd}[/red]")
        console.print("[red]Usage:[/red] /workspace list | new <name> | switch <name> | delete <name>")
