"""
rag/cli.py — Motif interactive REPL and one-shot entry point.

Entry point registered in pyproject.toml:
    motif = "rag.cli:main"

Modes:
    Interactive (default):  `motif`
        Launches a prompt_toolkit REPL. Models stay loaded between queries.
        Plain text → query pipeline. Slash commands → command handlers.
        Session history persisted to ~/.ragdb/history.json on exit.

    One-shot (scripting):   `motif ask "query"` / `motif ingest ./docs`
        Executes a single command and exits. No REPL, no session.

Thread pool env vars are set at module import time BEFORE numpy/onnxruntime/
numexpr are imported. Setting them after import has no effect.
"""
from __future__ import annotations

# ── Thread pool limits — MUST be set before numpy/onnxruntime/numexpr import ─
import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
import platform
import subprocess
import shlex
from dataclasses import dataclass, field
from rich.panel import Panel
from rich.table import Table
from rich import box

from rag import __version__
from rag.config import load_config, RAGConfig
from rag.session import Session
from rag.commands import get_command, SLASH_COMMANDS, COMMAND_DESCRIPTIONS

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Welcome Screen
# ─────────────────────────────────────────────────────────────────────────────

def _render_welcome(config: RAGConfig, session: Session) -> None:
    """Render the startup welcome panel with system info and session state."""

    tier_label = config.resolved_tier
    backend_label = getattr(config.hardware, "backend", "cpu").upper()
    llm_name = Path(config.models.llm_path).stem
    db_root = config.db_root
    cwd = Path.cwd()

    # Try to get index stats (returns None if no index yet)
    chunk_count, doc_count = _get_index_stats(config)

    # Build info lines
    info_lines: list[str] = [
        f"[bold white]Motif[/bold white] [dim]v{__version__}[/dim]",
        "",
        f"  Tier    [bold]{tier_label}[/bold]  "
        f"[dim]|[/dim]  {llm_name}  "
        f"[dim]({backend_label})[/dim]",
    ]

    if chunk_count is not None:
        info_lines.append(
            f"  Index   [bold]{chunk_count:,}[/bold] chunks  "
            f"[dim]|[/dim]  {doc_count:,} documents"
        )
    else:
        info_lines.append(
            "  Index   [dim]none — run [bold]/ingest PATH[/bold] to add documents[/dim]"
        )

    info_lines += [
        f"  Dir     [dim]{cwd}[/dim]",
        "",
    ]

    # Cache warning — shown when query caching is enabled
    if getattr(config.storage, "query_cache_enabled", False):
        info_lines.append(
            "  [yellow]Query caching ON[/yellow] — queries stored at "
            f"[dim]{config.db_root}/query_cache.db[/dim]"
        )
        info_lines.append("")

    # Session history state
    if session.turn_count > 0 and session.last_query:
        truncated = (session.last_query[:60] + "…") if len(session.last_query) > 60 else session.last_query
        info_lines += [
            f"  Resuming previous session — "
            f"[bold]{session.turn_count}[/bold] exchange"
            f"{'s' if session.turn_count != 1 else ''}",
            f"  Last: [italic dim]\"{truncated}\"[/italic dim]",
            "  Type [bold]/new[/bold] to start fresh.",
        ]
    else:
        info_lines += [
            "  [dim]No previous session.[/dim]",
            "  Type [bold]/help[/bold] to see available commands.",
        ]

    body = "\n".join(info_lines)
    console.print(Panel(body, border_style="dim white", padding=(1, 2)))
    console.print()


def _get_index_stats(config: RAGConfig) -> tuple[int | None, int | None]:
    """Return (chunk_count, doc_count) or (None, None) if no index exists."""
    try:
        from rag.storage.chunk_store import ChunkStore
        store = ChunkStore(config)
        return store.count(), store.count_documents()
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Slash Command Router
# ─────────────────────────────────────────────────────────────────────────────

def _handle_slash_command(raw: str, session: Session, config: RAGConfig) -> None:
    """
    Parse and dispatch a slash command.

    Format:  /command [arg1 arg2 ...]
    Unknown commands print a friendly error and suggest /help.
    """
    import shlex
    parts = shlex.split(raw.strip(), posix=False)
    command_name = parts[0].lower()   # e.g. "/ingest"
    args = [arg.strip('"\'') for arg in parts[1:]]  # e.g. ["./docs", "-r"]

    handler = get_command(command_name)
    if handler is None:
        console.print(
            f"[red]Unknown command:[/red] {command_name}. "
            f"Type [bold]/help[/bold] for available commands."
        )
        return

    try:
        handler(args=args, session=session, config=config, console=console)
    except KeyboardInterrupt:
        console.print("\n[dim]Command interrupted.[/dim]")
    except Exception as exc:
        console.print(f"[red]Command error:[/red] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Query Handler (plain-text input)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_query(raw: str, session: Session, config: RAGConfig) -> None:
    """
    Parse inline modifiers from the query string and run the RAG pipeline.

    Inline modifier syntax (appended to the query):
        /file FILENAME    — restrict retrieval to this file
        /type TYPE        — restrict to document type (pdf, md, audio, image)
        /pages MIN-MAX    — restrict to page range
        /hyde             — enable HyDE query expansion (opt-in, adds ~2-5 s)
        /no-sources       — suppress citations in output

    Example:
        What does section 3 say? /file thesis.pdf /pages 20-40
        Explain attention mechanism /hyde
    """
    # Parse inline modifiers
    query, modifiers = _parse_query_modifiers(raw)

    if not query.strip():
        return

    # Check if pipeline is available (Phase 1+)
    try:
        from rag.pipeline import QueryPipeline
    except ImportError:
        console.print(
            "[yellow]Pipeline not yet implemented.[/yellow] "
            "Complete Phase 1 to enable query functionality.\n"
            "Run [bold]/ingest PATH[/bold] once the pipeline is ready."
        )
        return

    try:
        history_context = session.get_history_for_context(
            token_budget=config.generation.context_max_tokens,
            passage_tokens=0,   # pipeline will report actual passage tokens
        )

        pipeline = QueryPipeline(config)
        answer = pipeline.answer(
            query=query,
            history=history_context,
            file_filter=modifiers.get("file"),
            type_filter=modifiers.get("type"),
            page_range=modifiers.get("pages"),
            use_hyde=bool(modifiers.get("hyde", False)),   # opt-in HyDE
            show_sources=not modifiers.get("no-sources", False),
        )
        session.add_turn(query, answer.text)

    except KeyboardInterrupt:
        console.print("\n[dim]Generation interrupted.[/dim]")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")


def _parse_query_modifiers(raw: str) -> tuple[str, dict]:
    """
    Split inline /modifier flags from the end of a query string.

    Returns:
        (query_text, modifiers_dict)

    Example:
        "What is X? /file report.pdf /hyde"
        → ("What is X?", {"file": "report.pdf", "hyde": True})
    """
    tokens = raw.strip().split()
    query_tokens: list[str] = []
    modifiers: dict = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("/") and token != "/":
            key = token.lstrip("/")
            # Check if next token is a value (not another modifier or end)
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("/"):
                modifiers[key] = tokens[i + 1]
                i += 2
            else:
                modifiers[key] = True
                i += 1
        else:
            query_tokens.append(token)
            i += 1

    return " ".join(query_tokens), modifiers


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def _interactive_mode(no_prewarm: bool = False) -> None:
    """Launch the interactive prompt_toolkit REPL."""

    # Load config and session
    config = load_config()
    session = Session(config)
    session.load()

    # Ensure db_root exists
    os.makedirs(str(config.db_root), exist_ok=True)

    # Setup file logging
    import rag.logging_config
    rag.logging_config.setup(config)

    # ── Pre-warm models (Phase 4) ─────────────────────────────────────────────
    if not no_prewarm:
        try:
            from rag.warmup import prewarm_models
            prewarm_models(config, console=console)
        except Exception as exc:
            console.print(f"[yellow]Pre-warm skipped:[/yellow] {exc}")

    # Auto-calibrate threshold (will fast-path return if already done or index empty)
    from rag.retrieval.calibrate import calibrate_threshold
    calibrate_threshold(config, n_probes=10)

    # Welcome screen
    _render_welcome(config, session)

    # Tab completion for slash commands
    completer = WordCompleter(
        list(SLASH_COMMANDS.keys()) + ["exit", "quit"],
        sentence=True,
        match_middle=False,
    )

    # Key bindings: Ctrl+C at prompt exits gracefully
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _ctrl_c(event):
        raise KeyboardInterrupt()

    prompt_session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        completer=completer,
        key_bindings=bindings,
        enable_history_search=True,
    )

    # ── REPL loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            raw = prompt_session.prompt(HTML("<ansiblue>motif</ansiblue> <bold>&gt;</bold> "))
        except KeyboardInterrupt:
            # Ctrl+C at the prompt — save history and exit
            console.print("\n[dim]Saving session…[/dim]")
            session.save()
            console.print("[dim]Goodbye.[/dim]")
            break
        except EOFError:
            # Ctrl+D
            session.save()
            break

        raw = raw.strip()

        if not raw:
            continue

        if raw.lower() in ("exit", "quit"):
            session.save()
            console.print("[dim]Session saved. Goodbye.[/dim]")
            break

        if raw.startswith("/"):
            _handle_slash_command(raw, session, config)
        else:
            _handle_query(raw, session, config)


# ─────────────────────────────────────────────────────────────────────────────
# One-Shot Mode
# ─────────────────────────────────────────────────────────────────────────────

def _one_shot_mode(argv: list[str]) -> None:
    """
    Handle one-shot subcommands for scripting:
        motif ask "query"
        motif ingest ./docs
        motif setup [--tier T2]
        motif status
    """
    config = load_config()
    session = Session(config)

    subcommand = argv[0].lower()
    args = argv[1:]

    if subcommand == "ask":
        if not args:
            console.print("[red]Usage:[/red] motif ask \"your question\"")
            sys.exit(1)
        query = " ".join(args)
        _handle_query(query, session, config)

    elif subcommand in ("ingest", "remove", "sync", "status", "setup", "help"):
        # Route to the corresponding slash command handler
        slash = f"/{subcommand}"
        _handle_slash_command(f"{slash} {' '.join(args)}", session, config)

    else:
        console.print(f"[red]Unknown subcommand:[/red] {subcommand}")
        console.print("Run [bold]motif[/bold] (no arguments) to start the interactive session.")
        console.print("Run [bold]motif /help[/bold] to see all commands.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main entry point registered in pyproject.toml.

    Flags:
        --no-prewarm   Skip model pre-loading (first query will have cold-start latency).

    If additional arguments are provided, run in one-shot mode.
    Otherwise, launch the interactive REPL.
    """
    args = sys.argv[1:]

    # Handle --no-prewarm flag before routing
    no_prewarm = "--no-prewarm" in args
    args = [a for a in args if a != "--no-prewarm"]

    # Handle --hyde flag for one-shot mode (appends /hyde modifier)
    use_hyde = "--hyde" in args
    args = [a for a in args if a != "--hyde"]

    if args:
        if use_hyde and args[0].lower() == "ask":
            args.append("/hyde")
        _one_shot_mode(args)
    else:
        _interactive_mode(no_prewarm=no_prewarm)


if __name__ == "__main__":
    main()
