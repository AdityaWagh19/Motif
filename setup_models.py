"""
setup_models.py — Download models for a given hardware tier.

Usage:
    python setup_models.py --tier T2
    python setup_models.py --tier T3 --captioning
    python setup_models.py --verify

Also callable as the `motif setup` command (via rag/commands/setup.py).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, DownloadColumn, TextColumn

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"

# (repo_id, filename, local_name, tiers, size_label)
LLM_MODELS = [
    (
        "microsoft/Phi-3.5-mini-instruct-GGUF",
        "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        {"T1"},
        "2.2 GB",
    ),
    (
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q4_k_m.gguf",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        {"T2", "T3"},
        "4.2 GB",
    ),
]

EMBED_MODELS = [
    (
        "nomic-ai/nomic-embed-text-v1.5-ONNX",
        None,                          # snapshot (whole repo)
        "nomic-embed-text-v1.5",
        {"T1", "T2", "T3"},
        "274 MB",
    ),
]

RERANKER_MODELS = [
    (
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        None,
        "MiniLM-L12-v2",
        {"T1", "T2"},
        "134 MB",
    ),
    (
        "BAAI/bge-reranker-base",
        None,
        "bge-reranker-base",
        {"T3"},
        "280 MB",
    ),
]

WHISPER_MODELS = [
    (
        "ggerganov/whisper.cpp",
        "ggml-tiny-q5_1.bin",
        "ggml-tiny-q5_1.bin",
        {"T1", "T2"},
        "75 MB",
    ),
    (
        "ggerganov/whisper.cpp",
        "ggml-small-q5_1.bin",
        "ggml-small-q5_1.bin",
        {"T3"},
        "244 MB",
    ),
]

CAPTIONING_MODELS = [
    (
        "vikhyatk/moondream2",
        None,
        "moondream2",
        {"T3"},
        "~900 MB",
    ),
]

# Total size reference per tier
TIER_SIZES = {"T1": "2.8 GB", "T2": "4.9 GB", "T3": "5.2 GB"}


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(repo_id: str, filename: str, local_name: str, size_label: str) -> Path:
    """Download a single file from HuggingFace Hub to models/."""
    dest = MODELS_DIR / local_name
    if dest.exists():
        console.print(f"  [dim]skip[/dim]  {local_name} (already downloaded)")
        return dest

    console.print(f"  [cyan]down[/cyan]  {local_name} ({size_label})")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
    )
    # Rename to local_name if different
    actual = Path(path)
    target = MODELS_DIR / local_name
    if actual != target:
        actual.rename(target)
    return target


def _download_snapshot(repo_id: str, local_name: str, size_label: str) -> Path:
    """Download a full HuggingFace repo snapshot to models/<local_name>/."""
    dest = MODELS_DIR / local_name
    if dest.exists() and any(dest.iterdir()):
        console.print(f"  [dim]skip[/dim]  {local_name}/ (already downloaded)")
        return dest

    console.print(f"  [cyan]down[/cyan]  {local_name}/ ({size_label})")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        local_dir_use_symlinks=False,
    )
    return dest


def _download_model(entry: tuple, tier: str) -> bool:
    """Download a model entry if it belongs to the given tier. Returns True if downloaded."""
    repo_id, filename, local_name, tiers, size_label = entry
    if tier not in tiers:
        return False

    if filename:
        _download_file(repo_id, filename, local_name, size_label)
    else:
        _download_snapshot(repo_id, local_name, size_label)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────

def _verify(tier: str, captioning: bool) -> None:
    """Check which models are present and print a verification table."""
    from rich.table import Table
    from rich import box

    all_models = LLM_MODELS + EMBED_MODELS + RERANKER_MODELS + WHISPER_MODELS
    if captioning:
        all_models += CAPTIONING_MODELS

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    table.add_column("Model", style="dim")
    table.add_column("Size")
    table.add_column("Status")

    all_ok = True
    for repo_id, filename, local_name, tiers, size_label in all_models:
        if tier not in tiers:
            continue
        path = MODELS_DIR / local_name
        exists = path.exists() and (path.is_file() or (path.is_dir() and any(path.iterdir())))
        status = "[green]ok[/green]" if exists else "[red]missing[/red]"
        if not exists:
            all_ok = False
        table.add_row(local_name, size_label, status)

    console.print(table)
    if all_ok:
        console.print(f"\n[green]All models present for Tier {tier}.[/green]")
        console.print("Run [bold]motif[/bold] to start.\n")
    else:
        console.print(f"\n[yellow]Some models missing.[/yellow] Run: [bold]motif setup --tier {tier}[/bold]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Motif models for your hardware tier."
    )
    parser.add_argument(
        "--tier",
        choices=["T1", "T2", "T3"],
        default=None,
        help="Hardware tier (default: auto-detect from GPU VRAM)",
    )
    parser.add_argument(
        "--captioning",
        action="store_true",
        help="Also download moondream2 Q4 for image captioning (T3 opt-in, ~900 MB)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check which models are present without downloading",
    )
    args = parser.parse_args()

    # Auto-detect tier if not specified
    if args.tier is None:
        from rag.config import detect_hardware_tier
        args.tier = detect_hardware_tier()
        console.print(f"[dim]Auto-detected tier:[/dim] [bold]{args.tier}[/bold]")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        _verify(args.tier, args.captioning)
        return

    console.print(
        f"\n[bold]Downloading models for Tier {args.tier}[/bold] "
        f"({TIER_SIZES.get(args.tier, '?')} total)\n"
    )

    for entry in LLM_MODELS + EMBED_MODELS + RERANKER_MODELS + WHISPER_MODELS:
        try:
            _download_model(entry, args.tier)
        except Exception as exc:
            console.print(f"  [red]fail[/red]  {entry[2]}: {exc}")

    if args.captioning:
        console.print("\n[dim]Downloading image captioning model (moondream2)…[/dim]")
        for entry in CAPTIONING_MODELS:
            try:
                _download_model(entry, args.tier)
            except Exception as exc:
                console.print(f"  [red]fail[/red]  {entry[2]}: {exc}")

    console.print()
    _verify(args.tier, args.captioning)


if __name__ == "__main__":
    main()
