"""rag/commands/help.py — /help command."""
from __future__ import annotations

from rich import box
from rich.table import Table


def handle_help(args, session, config, console) -> None:
    """Print all available slash commands with descriptions."""
    from rag.commands import COMMAND_DESCRIPTIONS, COMMAND_EXAMPLES

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    table.add_column("Command", style="accent_bold", no_wrap=True)
    table.add_column("Description", style="dim")
    table.add_column("Example", style="muted_italic")

    for cmd, desc in COMMAND_DESCRIPTIONS.items():
        example = COMMAND_EXAMPLES.get(cmd, "")
        table.add_row(cmd, desc, example)

    console.print()
    console.print(table)

    console.print(
        "\n[accent_bold]Query Modifiers[/accent_bold] [dim](append to any question):[/dim]\n"
        "  [accent]/file FILENAME[/accent]    Restrict to a specific document\n"
        "  [accent]/type TYPE[/accent]        Restrict to document type (pdf, md, audio, image)\n"
        "  [accent]/pages MIN-MAX[/accent]    Restrict to a page range\n"
        "  [accent]/hyde[/accent]              Enable HyDE query expansion\n"
        "  [accent]/no-sources[/accent]       Suppress citations in output\n\n"
        "[dim]Example query with modifiers:[/dim]\n"
        "  [muted_italic]What is section 3 about? /file thesis.pdf /pages 10-25[/muted_italic]"
    )
