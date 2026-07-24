"""
rag/cli.py — Motif interactive REPL and one-shot entry point.

Entry point registered in pyproject.toml:
    motif = "rag.cli:main"

Modes:
    Interactive (default):  `motif`
        Launches a prompt_toolkit REPL. Models stay loaded between queries.
        Plain text → query pipeline. Slash commands → command handlers.
        Session history persisted to storage on exit.

    One-shot (scripting):   `motif ask "query"` / `motif ingest ./docs`
        Executes a single command and exits. No REPL, no session.
"""
from __future__ import annotations

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
import logging
import shlex
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

warnings.filterwarnings("ignore")

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.panel import Panel

from rag import __version__
from rag.commands import SLASH_COMMANDS, get_command
from rag.config import RAGConfig, load_config
from rag.errors import humanize_error
from rag.session import Session
from rag.theme import console

if TYPE_CHECKING:
    from rag.pipeline import QueryPipeline


def setup_cli_logging() -> None:
    """Redirect ALL logs to ~/.motif/logs/motif.log. No output reaches the terminal."""
    try:
        log_dir = Path.home() / ".motif" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "motif.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    except Exception:
        logging.getLogger().addHandler(logging.NullHandler())

    for noisy in ["ppocr", "rag.retrieval.calibrate", "qdrant_client", "onnxruntime", "urllib3", "httpx", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.ERROR)


setup_cli_logging()
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Welcome Screen
# ─────────────────────────────────────────────────────────────────────────────

def _render_welcome(config: RAGConfig, session: Session) -> None:
    """Render the startup welcome panel with clean system info and session state."""

    accel_name = getattr(config.hardware, "backend", "cpu").upper()
    accel_label = f"GPU Accelerated ({accel_name})" if accel_name != "CPU" else "CPU Mode"
    llm_stem = Path(config.models.llm_path).stem
    if "qwen" in llm_stem.lower():
        llm_label = "Qwen2.5 7B"
    elif "llama" in llm_stem.lower():
        llm_label = "Llama 3.1 8B"
    elif "mistral" in llm_stem.lower():
        llm_label = "Mistral 7B"
    else:
        llm_label = llm_stem.split("-")[0]

    data_dir = config.db_root

    chunk_count, doc_count = _get_index_stats(config)
    index_str = f"{doc_count:,} documents ({chunk_count:,} passages)" if chunk_count is not None and doc_count else "0 documents (run /ingest)"

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
        f"  [accent_bold]Motif[/accent_bold] [subtle]v{__version__}[/subtle]  [subtle]|[/subtle]  Offline Local RAG AI Assistant",
        f"  Model   [bold]{llm_label}[/bold]  [subtle]|[/subtle]  Mode: {accel_label}",
        f"  Index   [bold]{index_str}[/bold]",
        f"  Data    [subtle]{data_dir}[/subtle]",
    ]

    if session.turn_count > 0 and session.last_query:
        truncated = (session.last_query[:50] + "…") if len(session.last_query) > 50 else session.last_query
        info_lines.append(f"  Session [accent_bold]{session.turn_count}[/accent_bold] turns [subtle](Last: \"{truncated}\")[/subtle]")

    cat_art = (
        "\n"
        "[accent]          ████          ████          [/accent]\n"
        "[accent]        ██████████████████████        [/accent]\n"
        "[accent]  ▀▀▀▀  ████▄▄██▀▀▀▀▀▀██▄▄████  ▀▀▀▀  [/accent]\n"
        "[accent]  ▀▀▀▀  ██████████▄▄██████████  ▀▀▀▀  [/accent]\n"
        "[accent]        ██████████████████████▄▄████  [/accent]  [dim]\"Ready to search your local documents. Type /help for options.\"[/dim]\n"
        "[accent]        ████  ████  ████  ████        [/accent]"
    )
    
    info_lines.append(cat_art)
    console.print(Panel("\n".join(info_lines), border_style="structure", padding=(1, 2)))
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
    """
    try:
        parts = shlex.split(raw.strip(), posix=False)
    except Exception:
        parts = raw.strip().split()

    if not parts:
        return

    command_name = parts[0].lower()
    args = [arg.strip('"\'') for arg in parts[1:]]

    handler = get_command(command_name)
    if handler is None:
        import difflib
        matches = difflib.get_close_matches(command_name, SLASH_COMMANDS.keys(), n=1, cutoff=0.6)
        suggestion = f" Did you mean [accent_bold]{matches[0]}[/accent_bold]?" if matches else ""
        console.print(
            f"[error]Unknown command:[/error] {command_name}.{suggestion} "
            f"Type [accent_bold]/help[/accent_bold] for available commands."
        )
        return

    try:
        handler(args=args, session=session, config=config, console=console)
    except KeyboardInterrupt:
        console.print("\n[subtle]^C [Command cancelled][/subtle]")
    except Exception as exc:
        log.debug("Slash command exception during %s: %s", command_name, exc, exc_info=True)
        console.print(f"[error]Could not run command:[/error] {humanize_error(exc)}")


# ─────────────────────────────────────────────────────────────────────────────
# Query Handler (plain-text input)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_query(raw: str, session: Session, config: RAGConfig, pipeline: QueryPipeline | None = None) -> None:
    """
    Parse inline modifiers from the query string and run the RAG pipeline.
    """
    query, modifiers = _parse_query_modifiers(raw)

    if not query.strip():
        return

    try:
        from rag.pipeline import QueryPipeline
    except ImportError as exc:
        console.print(f"[error]Query service unavailable:[/error] {humanize_error(exc)}")
        return

    close_pipeline_on_finish = False
    if pipeline is None:
        pipeline = QueryPipeline(config)
        close_pipeline_on_finish = True

    try:
        history_context = session.get_history_for_context(
            token_budget=config.generation.context_max_tokens,
            passage_tokens=0,
        )

        answer = pipeline.answer(
            query=query,
            history=history_context,
            file_filter=modifiers.get("file"),
            type_filter=modifiers.get("type"),
            page_range=modifiers.get("pages"),
            use_hyde=bool(modifiers.get("hyde", False)),
            show_sources=not modifiers.get("no-sources", False),
        )
        session.add_turn(query, answer.text)
        console.print()  # Visual spacing between turn and next prompt

    except KeyboardInterrupt:
        console.print("\n[subtle]^C [Query cancelled][/subtle]")
    except Exception as exc:
        log.debug("Query error for input '%s': %s", raw, exc, exc_info=True)
        console.print(f"[error]Could not answer query:[/error] {humanize_error(exc)}")
    finally:
        if close_pipeline_on_finish and hasattr(pipeline, "close"):
            pipeline.close()


def _parse_query_modifiers(raw: str) -> tuple[str, dict]:
    """
    Split inline /modifier flags from the end of a query string using shlex.
    """
    try:
        tokens = shlex.split(raw.strip(), posix=False)
    except Exception:
        tokens = raw.strip().split()

    query_tokens: list[str] = []
    modifiers: dict = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("/") and token != "/":
            key = token.lstrip("/")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("/"):
                val = tokens[i + 1].strip("\"'")
                modifiers[key] = val
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

    config = load_config()
    session = Session(config)
    session.load()

    os.makedirs(str(config.db_root), exist_ok=True)

    import rag.logging_config
    rag.logging_config.setup(config)

    try:
        from rag.storage.reconciler import StorageReconciler
        StorageReconciler.reconcile_all(config)
    except Exception as exc:
        log.debug("Storage reconciliation notice: %s", exc)

    if not no_prewarm:
        try:
            from rag.warmup import prewarm_models
            prewarm_models(config, console=console)
        except Exception as exc:
            log.debug("Pre-warm notice: %s", exc)

    from rag.retrieval.calibrate import calibrate_threshold
    calibrate_threshold(config, n_probes=10)

    _render_welcome(config, session)

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

    bindings = KeyBindings()

    @bindings.add("c-c")
    def _ctrl_c(event):
        raise KeyboardInterrupt()

    # Pre-cache toolbar label strings outside toolbar render closure
    workspace = config.db_root.name
    backend = getattr(config.hardware, "backend", "cpu").upper()
    mode_label = "GPU Accelerated" if backend != "CPU" else "CPU Mode"
    llm_stem = Path(config.models.llm_path).stem
    if "qwen" in llm_stem.lower():
        model_label = "Qwen2.5 7B"
    elif "llama" in llm_stem.lower():
        model_label = "Llama 3.1 8B"
    elif "mistral" in llm_stem.lower():
        model_label = "Mistral 7B"
    else:
        model_label = llm_stem.split("-")[0]

    def get_bottom_toolbar():
        return HTML(
            f' <style fg="#6b7280">Workspace:</style> <style fg="#FF2E93">{workspace}</style>  <style fg="#6b7280">|</style>  '
            f'<style fg="#6b7280">Model:</style> <style fg="#FF2E93">{model_label}</style>  <style fg="#6b7280">|</style>  '
            f'<style fg="#6b7280">Mode:</style> <style fg="#FF2E93">{mode_label}</style> '
        )

    custom_style = Style.from_dict({
        "bottom-toolbar": "noreverse bg:default",
    })

    from prompt_toolkit.history import FileHistory
    history_file = config.db_root / ".prompt_history"

    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
        completer=completer,
        key_bindings=bindings,
        enable_history_search=True,
        bottom_toolbar=get_bottom_toolbar,
        style=custom_style,
    )

    from rag.pipeline import QueryPipeline
    pipeline = QueryPipeline(config)
    current_workspace = config.storage.workspace

    try:
        while True:
            try:
                raw = prompt_session.prompt(HTML('<b><style fg="#FF2E93">motif ❯</style></b> '))
            except KeyboardInterrupt:
                session.save()
                console.print("\n[subtle]Goodbye.[/subtle]")
                break
            except EOFError:
                session.save()
                console.print("\n[subtle]Goodbye.[/subtle]")
                break

            raw = raw.strip()

            if not raw:
                continue

            if raw.lower() in ("exit", "quit"):
                session.save()
                console.print("[subtle]Goodbye.[/subtle]")
                break

            if raw.startswith("/"):
                _handle_slash_command(raw, session, config)
                if config.storage.workspace != current_workspace:
                    log.info("Workspace changed from %s to %s", current_workspace, config.storage.workspace)
                    if hasattr(pipeline, "close"):
                        pipeline.close()
                    pipeline = QueryPipeline(config)
                    current_workspace = config.storage.workspace
            else:
                _handle_query(raw, session, config, pipeline=pipeline)
    finally:
        if hasattr(pipeline, "close"):
            pipeline.close()


# ─────────────────────────────────────────────────────────────────────────────
# One-Shot Mode
# ─────────────────────────────────────────────────────────────────────────────

def _one_shot_mode(argv: list[str]) -> None:
    """
    Handle one-shot subcommands for scripting.
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
        from rag.pipeline import QueryPipeline
        pipeline = QueryPipeline(config)
        try:
            _handle_query(query, session, config, pipeline=pipeline)
        finally:
            if hasattr(pipeline, "close"):
                pipeline.close()

    elif subcommand in ("ingest", "remove", "sync", "status", "setup", "help", "--help", "-h"):
        if subcommand in ("--help", "-h"):
            subcommand = "help"
        slash = f"/{subcommand}"
        _handle_slash_command(f"{slash} {' '.join(args)}", session, config)

    else:
        console.print(f"[error]Unknown subcommand:[/error] {subcommand}")
        console.print("Run [accent_bold]motif[/accent_bold] (no arguments) to start interactive mode.")
        console.print("Run [accent_bold]motif /help[/accent_bold] to view commands.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main entry point registered in pyproject.toml.
    """
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        from rag import __version__
        console.print(f"\n[accent_bold]Motif[/accent_bold] [subtle]v{__version__}[/subtle] — Offline Local RAG AI Assistant\n")
        console.print("[accent_bold]Usage:[/accent_bold]")
        console.print("  [bold]motif[/bold]                     Start interactive REPL session")
        console.print("  [bold]motif ask \"<query>\"[/bold]       Run a single query and print answer")
        console.print("  [bold]motif ingest <path>[/bold]       Ingest files or directories")
        console.print("  [bold]motif setup [--tier T2][/bold]    Verify/download AI models")
        console.print("  [bold]motif status[/bold]              Display system & index status")
        console.print("  [bold]motif sync [path][/bold]         Re-index updated files")
        console.print("  [bold]motif remove <path>[/bold]       Remove file from vector store")
        console.print("  [bold]motif --version[/bold]           Print version information\n")
        sys.exit(0)

    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    if "--version" in args:
        from rag import __version__
        console.print(f"Motif v{__version__}")
        sys.exit(0)

    verbose = "--verbose" in args
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    args = [a for a in args if a != "--verbose"]

    no_prewarm = "--no-prewarm" in args
    args = [a for a in args if a != "--no-prewarm"]

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
