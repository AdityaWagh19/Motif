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

from rag.config import load_config, migrate_if_needed

migrate_if_needed()
_boot_config = load_config()
_threads = str(_boot_config.llm.threads)

os.environ.setdefault("NUMEXPR_MAX_THREADS", _threads)
os.environ.setdefault("OMP_NUM_THREADS", _threads)
os.environ.setdefault("MKL_NUM_THREADS", _threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _threads)
# ─────────────────────────────────────────────────────────────────────────────
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("ppocr").setLevel(logging.ERROR)


import shlex
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.panel import Panel

from rag import __version__
from rag.commands import get_command
from rag.config import RAGConfig, load_config
from rag.session import Session
from rag.theme import console

# ─────────────────────────────────────────────────────────────────────────────
# Welcome Screen
# ─────────────────────────────────────────────────────────────────────────────

def _render_welcome(config: RAGConfig, session: Session) -> None:
    """Render the startup welcome panel with system info, Mochi mascot, and session state."""

    tier_label = config.resolved_tier
    backend_label = getattr(config.hardware, "backend", "cpu").upper()
    llm_name = Path(config.models.llm_path).stem
    cwd = Path.cwd()

    # Try to get index stats (returns None if no index yet)
    chunk_count, doc_count = _get_index_stats(config)
    index_str = f"{chunk_count:,} chunks | {doc_count:,} docs" if chunk_count is not None else "0 docs (run /ingest)"

    logo_art = (
        "[accent_bold]  ███╗   ███╗ ██████╗ ████████╗██╗███████╗[/accent_bold]\n"
        "[accent_bold]  ████╗ ████║██╔═══██╗╚══██╔══╝██║██╔════╝[/accent_bold]\n"
        "[accent_bold]  ██╔████╔██║██║   ██║   ██║   ██║███████╗[/accent_bold]\n"
        "[accent_bold]  ██║╚██╔╝██║██║   ██║   ██║   ██║██╔════╝[/accent_bold]\n"
        "[accent_bold]  ██║ ╚═╝ ██║╚██████╔╝   ██║   ██║██║     [/accent_bold]\n"
        "[accent_bold]  ╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝╚═╝     [/accent_bold]"
    )

    info_lines: list[str] = [
        logo_art,
        "",
        f"  [accent_bold]Mochi[/accent_bold] [structure]v{__version__}[/structure]  [structure]|[/structure]  Local RAG AI Assistant",
        f"  Tier    [bold]{tier_label}[/bold]  [structure]|[/structure]  {llm_name}  [structure]({backend_label})[/structure]",
        f"  Index   [bold]{index_str}[/bold]",
        f"  Dir     [structure]{cwd}[/structure]",
    ]

    # Session history state
    if session.turn_count > 0 and session.last_query:
        truncated = (session.last_query[:50] + "…") if len(session.last_query) > 50 else session.last_query
        info_lines.append(f"  Session [accent_bold]{session.turn_count}[/accent_bold] turns [structure](Last: \"{truncated}\")[/structure]")

    cat_sleep = (
        "\n"
        "[accent]          ████          ████          [/accent]\n"
        "[accent]        ██████████████████████        [/accent]\n"
        "[accent]  ▀▀▀▀  ████████▀▀▀▀▀▀████████  ▀▀▀▀  [/accent]\n"
        "[accent]  ▀▀▀▀  ██████████▄▄██████████  ▀▀▀▀  [/accent]\n"
        "[accent]        ██████████████████████▄▄████  [/accent]  [dim]zzz...[/dim]\n"
        "[accent]        ████  ████  ████  ████        [/accent]"
    )

    cat_awake = (
        "\n"
        "[accent]          ████          ████          [/accent]\n"
        "[accent]        ██████████████████████        [/accent]\n"
        "[accent]  ▀▀▀▀  ████▄▄██▀▀▀▀▀▀██▄▄████  ▀▀▀▀  [/accent]\n"
        "[accent]  ▀▀▀▀  ██████████▄▄██████████  ▀▀▀▀  [/accent]\n"
        "[accent]        ██████████████████████▄▄████  [/accent]  [dim]\"Ready to search your local documents. Type /help for options.\"[/dim]\n"
        "[accent]        ████  ████  ████  ████        [/accent]"
    )
    
    info_lines.append(cat_sleep)
    
    import time
    from rich.live import Live
    with Live(Panel("\n".join(info_lines), border_style="structure", padding=(1, 2)), console=console, refresh_per_second=10, transient=False) as live:
        time.sleep(0.6)
        info_lines[-1] = cat_awake
        live.update(Panel("\n".join(info_lines), border_style="structure", padding=(1, 2)))
        
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
    parts = shlex.split(raw.strip(), posix=False)
    command_name = parts[0].lower()   # e.g. "/ingest"
    args = [arg.strip('"\'') for arg in parts[1:]]  # e.g. ["./docs", "-r"]

    handler = get_command(command_name)
    if handler is None:
        console.print(
            f"[error]Unknown command:[/error] {command_name}. "
            f"Type [accent_bold]/help[/accent_bold] for available commands."
        )
        return

    try:
        handler(args=args, session=session, config=config, console=console)
    except KeyboardInterrupt:
        console.print("\n[structure]Command interrupted.[/structure]")
    except Exception as exc:
        console.print(f"[error]Command error:[/error] {exc}")


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
            "[warning]Pipeline not yet implemented.[/warning] "
            "Complete Phase 1 to enable query functionality.\n"
            "Run [accent_bold]/ingest PATH[/accent_bold] once the pipeline is ready."
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
        console.print()
        console.print("[structure]Generation interrupted.[/structure]")
    except Exception as exc:
        console.print(f"[error]Error:[/error] {exc}")


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
            console.print(f"[warning]Pre-warm skipped:[/warning] {exc}")

    # Auto-calibrate threshold (will fast-path return if already done or index empty)
    from rag.retrieval.calibrate import calibrate_threshold
    calibrate_threshold(config, n_probes=10)

    # Welcome screen
    _render_welcome(config, session)

    # Smart Autocomplete
    def get_workspaces():
        ws_dir = config.db_root.parent
        if ws_dir.exists():
            return [d.name for d in ws_dir.iterdir() if d.is_dir()]
        return []

    completer = NestedCompleter.from_nested_dict({
        "/ingest": PathCompleter(expanduser=True),
        "/sync": PathCompleter(expanduser=True),
        "/workspace": {
            "list": None,
            "new": None,
            "switch": WordCompleter(get_workspaces),
            "delete": WordCompleter(get_workspaces),
        },
        "/remove": None,
        "/status": None,
        "/clear": None,
        "/new": None,
        "/setup": None,
        "/help": None,
        "/exit": None,
        "/quit": None,
        "exit": None,
        "quit": None,
    })

    # Key bindings: Ctrl+C at prompt exits gracefully
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _ctrl_c(event):
        raise KeyboardInterrupt()

    def get_bottom_toolbar():
        workspace = config.db_root.name
        tier = config.resolved_tier
        backend = getattr(config.hardware, "backend", "cpu").upper()
        model = Path(config.models.llm_path).stem
        
        return HTML(
            f' <style fg="#6b7280">Workspace:</style> <style fg="#FF2E93">{workspace}</style>  <style fg="#6b7280">|</style>  '
            f'<style fg="#6b7280">Model:</style> <style fg="#FF2E93">{model}</style>  <style fg="#6b7280">|</style>  '
            f'<style fg="#6b7280">HW:</style> <style fg="#FF2E93">{tier}</style> <style fg="#6b7280">({backend})</style> '
        )

    custom_style = Style.from_dict({
        "bottom-toolbar": "noreverse bg:default",
    })

    prompt_session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        completer=completer,
        key_bindings=bindings,
        enable_history_search=True,
        bottom_toolbar=get_bottom_toolbar,
        style=custom_style,
    )

    # ── REPL loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            raw = prompt_session.prompt(HTML('<b><style fg="#FF2E93">motif ❯</style></b> '))
        except KeyboardInterrupt:
            # Ctrl+C at the prompt — save history and exit
            console.print("\n[structure]Saving session…[/structure]")
            session.save()
            console.print("[structure]Goodbye.[/structure]")
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
            console.print("[structure]Session saved. Goodbye.[/structure]")
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
            console.print("[error]Usage:[/error] motif ask \"your question\"")
            sys.exit(1)
        query = " ".join(args)
        _handle_query(query, session, config)

    elif subcommand in ("ingest", "remove", "sync", "status", "setup", "help", "--help", "-h"):
        if subcommand in ("--help", "-h"):
            subcommand = "help"
        # Route to the corresponding slash command handler
        slash = f"/{subcommand}"
        _handle_slash_command(f"{slash} {' '.join(args)}", session, config)

    else:
        console.print(f"[error]Unknown subcommand:[/error] {subcommand}")
        console.print("Run [accent_bold]motif[/accent_bold] (no arguments) to start the interactive session.")
        console.print("Run [accent_bold]motif /help[/accent_bold] to see all commands.")
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

    if "--help" in args or "-h" in args:
        from rag import __version__
        print(f"Motif v{__version__} — Local RAG AI Assistant")
        print("\nUsage:")
        print("  motif                     Start interactive REPL session")
        print("  motif ask \"<query>\"       Run a single query and print the answer")
        print("  motif ingest <path>       Ingest document files or directories")
        print("  motif setup [--tier T1|T2|T3]  Download/verify model files")
        print("  motif status              Display index and system status")
        print("  motif sync [path]         Re-index updated files")
        print("  motif remove <path>       Remove file from vector store")
        print("  motif --version           Print Motif version")
        print("  motif --help              Print this help overview")
        sys.exit(0)

    if "--version" in args:
        from rag import __version__
        print(f"Motif v{__version__}")
        sys.exit(0)

    verbose = "--verbose" in args
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    args = [a for a in args if a != "--verbose"]

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
