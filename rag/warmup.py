"""
rag/warmup.py — Pre-load all models at startup with progress reporting.

Called once from cli.py before the REPL begins.
Converts the 50 s first-query cold-start penalty into a transparent
startup phase with a Rich spinner progress bar.

Usage:
    from rag.warmup import prewarm_models
    prewarm_models(config, console=console)

Returns a dict of {model_name: load_time_ms} for logging.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

    from rag.config import RAGConfig

log = logging.getLogger(__name__)


def prewarm_models(config: RAGConfig, console: Console | None = None) -> dict:
    """
    Eagerly load embedder, reranker, and LLM before the REPL starts.

    Models are loaded in order:
      1. Embedder (nomic-embed-text-v1.5 ONNX) — used for ingestion + retrieval
      2. Reranker (MiniLM-L12-v2 / bge-reranker-base ONNX) — used after retrieval
      3. LLM (Phi-3.5-mini or Qwen2.5-7B GGUF) — used for generation

    On T1, the embedder is unloaded after ingestion to free RAM.
    This pre-warm does NOT change that behaviour — it just makes the
    initial load visible to the user rather than silent.

    Args:
        config:  Loaded RAGConfig with resolved_tier set.
        console: Rich Console for spinner output. None → silent.

    Returns:
        Dict of {model_name: load_time_ms}.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from rag.models.model_manager import get_model_manager

    manager = get_model_manager()
    timings: dict = {}

    llm_filename = config.models.llm_path.split("/")[-1]

    steps = [
        ("embedder", "Loading search engine...", lambda: manager.get_embedder(config)),
        ("reranker", "Loading reranker...", lambda: manager.get_reranker(config)),
        ("llm", "Loading AI model...", lambda: manager.get_llm(config)),
    ]

    warnings: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        for name, desc, loader in steps:
            task = progress.add_task(desc, total=None)
            t0 = time.monotonic()
            try:
                loader()
                elapsed_ms = round((time.monotonic() - t0) * 1000)
                timings[name] = elapsed_ms
            except FileNotFoundError as exc:
                elapsed_ms = round((time.monotonic() - t0) * 1000)
                timings[name] = elapsed_ms
                warnings.append(f"{name.capitalize()} model not found. Run `motif setup`.")
                log.warning("Pre-warm skipped %s: %s", name, exc)
            except Exception as exc:
                elapsed_ms = round((time.monotonic() - t0) * 1000)
                timings[name] = elapsed_ms
                warnings.append(f"{name.capitalize()} load notice: {exc}")
                log.error("Pre-warm error for %s: %s", name, exc)
            finally:
                progress.stop_task(task)

    total_ms = sum(timings.values())
    log.info("Pre-warm complete in %d ms: %s", total_ms, timings)

    if console:
        if warnings:
            for w in warnings:
                console.print(f"  [warning]•[/warning] {w}")
        else:
            console.print("[success]✓ AI engine initialized and ready.[/success]")

    return timings

    return timings
