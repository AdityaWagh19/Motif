"""rag/commands/help.py — /help command."""
from __future__ import annotations

from rich.table import Table
from rich import box


def handle_help(args, session, config, console) -> None:
    """Print all available slash commands with descriptions."""
    from rag.commands import SLASH_COMMANDS, COMMAND_DESCRIPTIONS

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Description", style="dim")

    for cmd, desc in COMMAND_DESCRIPTIONS.items():
        table.add_row(cmd, desc)

    table.add_section()
    table.add_row("exit / quit", "Save session and exit")

    console.print()
    console.print(table)

    console.print(
        "[dim]Query modifiers (append to any question):[/dim]\n"
        "  [cyan]/file FILENAME[/cyan]    Restrict to a specific document\n"
        "  [cyan]/type TYPE[/cyan]        Restrict to document type (pdf, md, audio, image)\n"
        "  [cyan]/pages MIN-MAX[/cyan]    Restrict to a page range\n"
        "  [cyan]/no-hyde[/cyan]          Skip HyDE query expansion\n"
        "  [cyan]/no-sources[/cyan]       Suppress citations\n"
    )
