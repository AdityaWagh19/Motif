"""rag/commands/status.py — /status command."""
from __future__ import annotations

from pathlib import Path

from rich import box
from rich.table import Table


def handle_status(args, session, config, console) -> None:
    """
    /status — Display index statistics and current system configuration.
    """
    # ── Index stats ───────────────────────────────────────────────────────────
    chunk_count: int | None = None
    doc_count: int | None = None
    bm25_count: int | None = None
    index_size_mb: float | None = None

    try:
        from rag.retrieval.bm25_index import BM25Index
        from rag.storage.chunk_store import ChunkStore
        with ChunkStore(config) as store:
            chunk_count = store.count()
            doc_count = store.count_documents()
        bm25_count = BM25Index.count_from_disk(config)
    except Exception:
        pass

    db_root = config.db_root
    if db_root.exists():
        total_bytes = sum(
            f.stat().st_size for f in db_root.rglob("*") if f.is_file()
        )
        index_size_mb = total_bytes / (1024 * 1024)

    # ── Build table ───────────────────────────────────────────────────────────
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Field", style="dim", no_wrap=True)
    table.add_column("Value", style="bold")

    # Index
    if chunk_count is not None:
        table.add_row("Knowledge Base", f"{doc_count} document{'s' if doc_count != 1 else ''} ({chunk_count:,} passages)")
    else:
        table.add_row("Knowledge Base", "[warning]Empty — run /ingest to add documents[/warning]")

    if index_size_mb is not None:
        table.add_row("Storage Used", f"{index_size_mb:.1f} MB")

    table.add_row("Session Turns", f"{session.turn_count} exchange{'s' if session.turn_count != 1 else ''}")
    table.add_section()

    # Hardware & Models (Human-Centered UI)
    from rag.config import resolve_model_path
    from rag.models.model_manager import get_model_manager
    manager = get_model_manager()
    llm_path = resolve_model_path(config.models.llm_path)

    llm_loaded = manager._llm is not None and getattr(manager._llm, "is_loaded", lambda: True)()
    status_str = "[green]Ready[/green]" if llm_loaded else "[dim]Standby[/dim]"

    backend_name = getattr(config.hardware, "backend", "cpu").upper()
    accel_str = f"GPU ({backend_name})" if backend_name != "CPU" else "CPU"

    table.add_row("Workspace", config.storage.workspace)
    table.add_row("AI Model", f"Qwen2.5 7B ({status_str})")
    table.add_row("Hardware Mode", f"{accel_str}")
    table.add_row("Search Mode", "Hybrid Semantic Search")
    table.add_section()

    table.add_row("Location", str(config.db_root))

    console.print()
    console.print(table)
