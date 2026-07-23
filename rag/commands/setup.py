"""rag/commands/setup.py — /setup command (model download)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def handle_setup(args, session, config, console) -> None:
    """
    /setup [--tier T1|T2|T3] [--captioning]

    Download models for your hardware tier.
    Delegates to setup_models.py in the project root.

    Options:
        --tier T1|T2|T3    Override auto-detected tier.
        --captioning        Also download moondream2 Q4 for T3 image captioning (~900 MB).
    """
    parser = argparse.ArgumentParser(prog="/setup", add_help=False)
    parser.add_argument("--tier", choices=["T1", "T2", "T3"], default=None)
    parser.add_argument("--captioning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    try:
        parsed = parser.parse_args(args)
    except SystemExit:
        console.print("[error]Usage:[/error] /setup [--tier T1|T2|T3] [--captioning]")
        return

    tier = parsed.tier or config.resolved_tier

    try:
        from rag.setup_models import main as setup_models_main
    except ImportError:
        console.print(
            "[error]setup_models module not found[/error].\n"
            "Ensure motif-rag is installed properly."
        )
        return

    old_argv = sys.argv.copy()
    sys.argv = ["setup_models.py", "--tier", tier]
    if parsed.captioning:
        sys.argv.append("--captioning")
    if parsed.dry_run:
        sys.argv.append("--dry-run")

    console.print(f"[structure]Running model download for Tier {tier}…[/structure]\n")
    try:
        setup_models_main()
    except Exception as exc:
        console.print(f"[error]Setup failed:[/error] {exc}")
    except KeyboardInterrupt:
        console.print("\n[structure]Setup interrupted.[/structure]")
    finally:
        sys.argv = old_argv
