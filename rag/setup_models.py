"""
rag/setup_models.py — Download models for a given hardware tier.

Usage:
    python -m rag.setup_models --tier T2
    python -m rag.setup_models --tier T3 --captioning
    python -m rag.setup_models --verify

Also callable as the `motif setup` CLI command.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# We intercept HuggingFace tqdm to use Rich
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
import huggingface_hub.utils
import huggingface_hub.file_download
import concurrent.futures

console = Console()

# Global UI for concurrent downloads
progress_ui = Progress(
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.description}"),
    BarColumn(),
    DownloadColumn(),
    TransferSpeedColumn(),
    console=console
)

class RichTqdm:
    def __init__(self, iterable=None, desc=None, total=None, leave=True, disable=False, **kwargs):
        self.disable = disable
        self.total = total or 100
        if iterable is not None:
            try:
                self.total = len(iterable)
            except Exception:
                pass
        self.desc = desc or "Downloading"
        self.iterable = iterable
        
        if not self.disable:
            self.task_id = progress_ui.add_task(self.desc, total=self.total)
            
    def update(self, n=1):
        if not self.disable:
            progress_ui.update(self.task_id, advance=n)
            
    def close(self):
        if not self.disable:
            progress_ui.stop_task(self.task_id)
            
    def set_description(self, desc):
        if not self.disable:
            progress_ui.update(self.task_id, description=desc)
            
    def __iter__(self):
        if self.iterable is None:
            return
        for obj in self.iterable:
            yield obj
            self.update(1)
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Apply patches so HuggingFace uses our Rich progress bars
huggingface_hub.utils.tqdm = RichTqdm
huggingface_hub.file_download.tqdm = RichTqdm

# ─────────────────────────────────────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────────────────────────────────────

from rag.config import _get_models_dir

MODELS_DIR: Path = _get_models_dir()

# (repo_id, filename, local_name, tiers, size_label)
LLM_MODELS = [
    (
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        {"T1"},
        "2.2 GB",
    ),
    (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        {"T2", "T3"},
        "4.2 GB",
    ),
]

EMBED_MODELS = [
    (
        "nomic-ai/nomic-embed-text-v1.5",
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

# WhisperX caches models dynamically via HuggingFace hub on first run (faster-whisper).
WHISPER_MODELS = []

CAPTIONING_MODELS = [
    (
        "vikhyatk/moondream2",
        None,
        "moondream2",
        {"T1", "T2", "T3"},
        "~900 MB",
    ),
]

# Total size reference per tier
TIER_SIZES = {"T1": "3.7 GB", "T2": "5.8 GB", "T3": "6.1 GB"}


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(repo_id: str, filename: str, local_name: str, size_label: str, dry_run: bool = False) -> Path:
    """Download a single file from HuggingFace Hub to models/."""
    dest = MODELS_DIR / local_name
    if dest.exists() and dest.stat().st_size > 0:
        progress_ui.console.print(f"  [dim]skip[/dim]  {local_name} (already downloaded)")
        return dest
        
    if dry_run:
        progress_ui.console.print(f"  [yellow]mock[/yellow]  {local_name} ({size_label})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()
        return dest

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(MODELS_DIR),
        token=False,
    )
    # Rename to local_name if different
    actual = Path(path)
    target = MODELS_DIR / local_name
    if actual != target:
        if actual.is_symlink():
            shutil.copy2(str(actual.resolve()), str(target))
        else:
            shutil.copy2(str(actual), str(target))
    progress_ui.console.print(f"  [green]ok[/green]    {local_name}")
    return target


def _get_nomic_onnx_patterns() -> list[str]:
    """
    Return the minimal allow_patterns for the nomic-embed-text ONNX download.
    """
    import platform as _plat
    machine = _plat.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        # Apple Silicon (M1/M2/M3) and ARM Linux
        return [
            "onnx/model_quantized_arm64.onnx",
            "onnx/model_quantized_arm64_data_0.onnx",
            "tokenizer*",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ]
    else:
        # x86_64 Windows, Linux, and Intel Mac
        return [
            "onnx/model_quantized.onnx",
            "tokenizer*",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ]


def _download_snapshot(repo_id: str, local_name: str, size_label: str, dry_run: bool = False) -> Path:
    """Download a full HuggingFace repo snapshot to models/<local_name>/."""
    dest = MODELS_DIR / local_name
    
    def is_real_download(p: Path) -> bool:
        return p.exists() and p.is_dir() and any(f.name != "mock_file.txt" for f in p.iterdir())
        
    if is_real_download(dest):
        progress_ui.console.print(f"  [dim]skip[/dim]  {local_name}/ (already downloaded)")
        return dest
        
    if dry_run:
        progress_ui.console.print(f"  [yellow]mock[/yellow]  {local_name}/ ({size_label})")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "mock_file.txt").touch()
        return dest

    snapshot_kwargs: dict = {}
    if "nomic" in repo_id:
        snapshot_kwargs["allow_patterns"] = _get_nomic_onnx_patterns()

    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        token=False,
        **snapshot_kwargs
    )
    progress_ui.console.print(f"  [green]ok[/green]    {local_name}/")
    return dest


def _download_model(entry: tuple, tier: str, dry_run: bool = False) -> bool:
    """Download a model entry if it belongs to the given tier. Returns True if downloaded."""
    repo_id, filename, local_name, tiers, size_label = entry
    if tier not in tiers:
        return False

    if filename:
        _download_file(repo_id, filename, local_name, size_label, dry_run=dry_run)
    else:
        _download_snapshot(repo_id, local_name, size_label, dry_run=dry_run)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────

def _verify(tier: str) -> None:
    """Check which models are present and print a verification table."""
    from rich import box
    from rich.table import Table

    all_models = LLM_MODELS + EMBED_MODELS + RERANKER_MODELS + WHISPER_MODELS + CAPTIONING_MODELS

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
        "--verify",
        action="store_true",
        help="Check which models are present without downloading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create empty mock files instead of downloading real models (for CI testing)",
    )
    args = parser.parse_args()

    # Auto-detect tier if not specified
    if args.tier is None:
        from rag.config import detect_hardware_tier
        args.tier = detect_hardware_tier()
        console.print(f"[dim]Auto-detected tier:[/dim] [bold]{args.tier}[/bold]")

    os.makedirs(str(MODELS_DIR), exist_ok=True)

    if args.verify:
        _verify(args.tier)
        return

    console.print(
        f"\n[bold]Downloading models for Tier {args.tier}[/bold] "
        f"({TIER_SIZES.get(args.tier, '?')} total)\n"
    )

    all_models = LLM_MODELS + EMBED_MODELS + RERANKER_MODELS + WHISPER_MODELS + CAPTIONING_MODELS
    
    # Sort models by size (GB > MB) so the largest GGUF downloads first
    def size_value(entry):
        label = entry[4]
        if "GB" in label:
            return float(label.replace("~", "").replace("GB", "").strip()) * 1000
        if "MB" in label:
            return float(label.replace("~", "").replace("MB", "").strip())
        return 0

    all_models.sort(key=size_value, reverse=True)

    with progress_ui:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for entry in all_models:
                futures.append(executor.submit(_download_model, entry, args.tier, args.dry_run))
                
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    progress_ui.console.print(f"  [red]fail[/red]  {exc}")

    console.print()
    _verify(args.tier)


if __name__ == "__main__":
    main()
