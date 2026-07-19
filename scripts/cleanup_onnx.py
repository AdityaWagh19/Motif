#!/usr/bin/env python3
"""
scripts/cleanup_onnx.py — Remove redundant ONNX variant files.

The HuggingFace snapshot_download() may fetch multiple ONNX variants that are
not needed on this platform (~390 MB of waste on x86_64 Linux).

Run this once after `motif setup` to reclaim disk space:
    python scripts/cleanup_onnx.py [--dry-run]

Disk savings (typical x86_64):
    model_fp16.onnx               120 MB   removed
    model_avx512.onnx             131 MB   removed
    model_avx2.onnx               131 MB   removed
    model_quantized_arm64.onnx    131 MB   removed
    ─────────────────────────────────────
    Total freed:                  ~390 MB  (keeps model_quantized.onnx @ 131 MB)

Usage:
    python scripts/cleanup_onnx.py             # ask for confirmation
    python scripts/cleanup_onnx.py --dry-run   # list only, do not delete
    python scripts/cleanup_onnx.py --yes       # skip confirmation prompt
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
NOMIC_DIR = MODELS_DIR / "nomic-embed-text-v1.5" / "onnx"

# x86_64: keep model_quantized.onnx, delete the rest
# arm64:  keep model_quantized_arm64.onnx, delete the rest
_X86_KEEP = {"model_quantized.onnx"}
_ARM_KEEP  = {"model_quantized_arm64.onnx", "model_quantized_arm64_data_0.onnx"}

_ALL_KNOWN_VARIANTS = {
    "model.onnx",
    "model_fp16.onnx",
    "model_quantized.onnx",
    "model_quantized_arm64.onnx",
    "model_quantized_arm64_data_0.onnx",
    "model_avx512.onnx",
    "model_avx2.onnx",
    "model_openvino.xml",
    "model_openvino.bin",
    "model_O3.onnx",
    "model_O4.onnx",
}


def _get_keep_set() -> frozenset[str]:
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return frozenset(_ARM_KEEP)
    return frozenset(_X86_KEEP)


def _format_size(path: Path) -> str:
    size = path.stat().st_size
    if size > 1_000_000:
        return f"{size / 1_000_000:.0f} MB"
    if size > 1_000:
        return f"{size / 1_000:.0f} KB"
    return f"{size} B"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove redundant ONNX variant files from models/nomic-embed-text-v1.5/onnx/",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files without deleting")
    parser.add_argument("--yes", "-y",  action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    keep = _get_keep_set()
    arch = platform.machine()
    print(f"Platform: {arch}")
    print(f"Keeping:  {', '.join(sorted(keep))}")
    print(f"ONNX dir: {NOMIC_DIR}")
    print()

    if not NOMIC_DIR.exists():
        print("ONNX directory not found. Run `motif setup` first.")
        sys.exit(0)

    to_delete: list[Path] = []
    for f in NOMIC_DIR.iterdir():
        if f.name in _ALL_KNOWN_VARIANTS and f.name not in keep:
            to_delete.append(f)

    if not to_delete:
        print("Nothing to clean up — all variant files already removed.")
        sys.exit(0)

    total_bytes = sum(f.stat().st_size for f in to_delete)
    total_mb = total_bytes / 1_000_000

    print(f"Files to delete ({total_mb:.0f} MB total):")
    for f in sorted(to_delete):
        print(f"  {_format_size(f):>8}  {f.name}")

    if args.dry_run:
        print("\n[dry-run] No files deleted.")
        sys.exit(0)

    if not args.yes:
        confirm = input(f"\nDelete {len(to_delete)} files ({total_mb:.0f} MB)? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    deleted = 0
    failed = 0
    for f in to_delete:
        try:
            f.unlink()
            print(f"  deleted  {f.name}")
            deleted += 1
        except OSError as exc:
            print(f"  ERROR    {f.name}: {exc}")
            failed += 1

    print(f"\nDone. Deleted {deleted} files ({total_mb:.0f} MB freed).")
    if failed:
        print(f"WARNING: {failed} file(s) could not be deleted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
