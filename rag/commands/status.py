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
        table.add_row("Documents", str(doc_count))
        table.add_row("Chunks", f"{chunk_count:,}")
        table.add_row("BM25 indexed", str(bm25_count) if bm25_count is not None else "—")
    else:
        table.add_row("Index", "[warning]No index found — run /ingest to add documents[/warning]")

    if index_size_mb is not None:
        table.add_row("Index size", f"{index_size_mb:.1f} MB")

    table.add_row("Session history", f"{session.turn_count} exchange{'s' if session.turn_count != 1 else ''}")
    table.add_section()

    # Hardware & Models (UX-09)
    from rag.models.model_manager import get_model_manager
    manager = get_model_manager()
    llm_path = Path(config.models.llm_path)
    llm_loaded = manager._llm is not None and getattr(manager._llm, "is_loaded", lambda: True)()
    embed_loaded = manager._embedder is not None and getattr(manager._embedder, "is_loaded", lambda: True)()
    rerank_loaded = manager._reranker is not None and getattr(manager._reranker, "is_loaded", lambda: True)()

    llm_str = "[green]✓ loaded[/green]" if llm_loaded else ("[dim]on disk[/dim]" if llm_path.exists() else "[red]✗ missing[/red]")
    embed_str = "[green]✓ loaded[/green]" if embed_loaded else "[dim]available[/dim]"
    rerank_str = "[green]✓ loaded[/green]" if rerank_loaded else "[dim]available[/dim]"

    table.add_row("Workspace", config.storage.workspace)
    table.add_row("Tier", config.resolved_tier)
    table.add_row("LLM", f"{llm_path.name} ({llm_str})")
    table.add_row("Embedder", f"nomic-embed-text-v1.5 ({embed_str})")
    table.add_row("Reranker", f"{Path(config.models.reranker).name} ({rerank_str})")
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
