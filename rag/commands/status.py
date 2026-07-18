"""rag/commands/status.py — /status command."""
from __future__ import annotations

from pathlib import Path

from rich.table import Table
from rich import box


def handle_status(args, session, config, console) -> None:
    """
    /status — Display index statistics and current system configuration.
    """
    # ── Index stats ───────────────────────────────────────────────────────────
    chunk_count: int | None = None
    doc_count: int | None = None
    index_size_mb: float | None = None

    try:
        from rag.storage.chunk_store import ChunkStore
        store = ChunkStore(config)
        chunk_count = store.count()
        doc_count = store.count_documents()
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
        table.add_row("Documents", str(doc_count))
        table.add_row("Chunks", f"{chunk_count:,}")
    else:
        table.add_row("Index", "[yellow]No index found — run /ingest to add documents[/yellow]")

    if index_size_mb is not None:
        table.add_row("Index size", f"{index_size_mb:.1f} MB")

    table.add_row("Session history", f"{session.turn_count} exchange{'s' if session.turn_count != 1 else ''}")
    table.add_section()

    # Hardware
    table.add_row("Tier", config.resolved_tier)
    table.add_row("LLM", Path(config.models.llm_path).name)
    table.add_row("GPU layers", str(config.llm.n_gpu_layers))
    table.add_row("Context window", f"{config.llm.ctx_size} tokens")
    table.add_row("Retrieval top-k", f"{config.retrieval.top_k_retrieval} → rerank → {config.retrieval.top_k_rerank}")
    table.add_row("Query expansion", config.retrieval.query_expansion)
    table.add_row("Semantic chunking", "on" if config.chunking.use_semantic else "off")
    table.add_section()

    # Storage
    table.add_row("DB root", str(config.db_root))

    console.print()
    console.print(table)
