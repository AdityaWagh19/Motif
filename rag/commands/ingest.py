"""rag/commands/ingest.py — /ingest command. (Phase 1 implementation pending)"""
from __future__ import annotations

import argparse
from pathlib import Path


def handle_ingest(args, session, config, console) -> None:
    """
    /ingest PATH [-r]

    Add documents from PATH to the knowledge base. Supported types: PDF, DOCX, MD,
    images, audio (based on installed parsers).

    Options:
        -r, --recursive   Recursively ingest all files in subdirectories.

    Phase 0: stub — prints a helpful message until the pipeline is implemented.
    Phase 1: wire to rag.ingestion.ingest_path().
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
        console.print("[red]Usage:[/red] /ingest PATH [-r]")
        return

    target = Path(parsed.path).expanduser().resolve()
    if not target.exists():
        console.print(f"[red]Path not found:[/red] {target}")
        return

    # ── Phase 1+ wiring ───────────────────────────────────────────────────────
    try:
        from rag.ingestion import ingest_path
        ingest_path(target, config=config, recursive=parsed.recursive, console=console)
    except ImportError:
        # Phase 0: pipeline not yet implemented
        console.print(
            f"[yellow]Ingestion pipeline not yet implemented[/yellow] "
            f"(Phase 1).\n"
            f"Target: [dim]{target}[/dim]  "
            f"Recursive: {'yes' if parsed.recursive else 'no'}"
        )
