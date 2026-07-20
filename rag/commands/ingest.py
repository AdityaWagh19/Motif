"""rag/commands/ingest.py — /ingest command handler."""
from __future__ import annotations

import argparse
from pathlib import Path


def handle_ingest(args, session, config, console) -> None:
    """
    /ingest PATH [-r]

    Ingest documents from PATH into the knowledge base.

    Supported types: .pdf, .md, .txt, .markdown
    Phase 4 adds: .docx
    Phase 5 adds: images, audio

    Options:
        -r, --recursive   Recursively ingest all files in subdirectories.
    """
    parser = argparse.ArgumentParser(prog="/ingest", add_help=False)
    parser.add_argument("path", nargs="?", help="Path to file or directory")
    parser.add_argument("-r", "--recursive", action="store_true")

    try:
        parsed = parser.parse_args(args)
    except SystemExit:
        console.print("[red]Usage:[/red] /ingest PATH [-r]")
        return

    if not parsed.path:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import PathCompleter
        from prompt_toolkit.formatted_text import HTML
        
        try:
            path_str = prompt(
                HTML("<ansiblue>?</ansiblue> Enter path to ingest: "),
                completer=PathCompleter(expanduser=True)
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return
            
        if not path_str:
            return
        parsed.path = path_str

    target = Path(parsed.path).expanduser().resolve()  # type: ignore[union-attr]
    if not target.exists():
        console.print(f"[red]Path not found:[/red] {target}")
        return

    from rag.ingestion import ingest_path

    console.print(f"\n[bold]Ingesting[/bold] {target}  recursive={parsed.recursive}\n")

    result = ingest_path(
        target,
        config=config,
        recursive=parsed.recursive,
        console=console,
    )

    console.print(
        f"\n[green]Done.[/green] "
        f"Files: [bold]{result.files_processed}[/bold]  "
        f"Chunks added: [bold]{result.chunks_added:,}[/bold]  "
        f"Skipped (unchanged): [bold]{result.files_skipped}[/bold]"
    )
    if result.errors:
        console.print(f"[yellow]Errors ({len(result.errors)}):[/yellow]")
        for err in result.errors:
            console.print(f"  [red]•[/red] {err}")
