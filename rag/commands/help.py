"""rag/commands/help.py — /help command."""
from __future__ import annotations

from rich.table import Table
from rich import box


def handle_help(args, session, config, console) -> None:
    """Print all available slash commands with descriptions."""
    from rag.commands import SLASH_COMMANDS, COMMAND_DESCRIPTIONS, COMMAND_EXAMPLES

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    table.add_column("Example", style="dim italic")

    for cmd, desc in COMMAND_DESCRIPTIONS.items():
        example = COMMAND_EXAMPLES.get(cmd, "")
        table.add_row(cmd, desc, example)

    console.print()
    console.print(table)

    console.print(
        "[structure]Query modifiers (append to any question):[/structure]\n"
        "  [cyan]/file FILENAME[/cyan]    Restrict to a specific document\n"
        "  [cyan]/type TYPE[/cyan]        Restrict to document type (pdf, md, audio, image)\n"
        "  [cyan]/pages MIN-MAX[/cyan]    Restrict to a page range\n"
        "  [cyan]/no-hyde[/cyan]          Skip HyDE query expansion\n"
        "  [cyan]/no-sources[/cyan]       Suppress citations\n"
    )
