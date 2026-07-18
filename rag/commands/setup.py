"""rag/commands/setup.py — /setup command (model download)."""
from __future__ import annotations

import argparse
import subprocess
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

    try:
        parsed = parser.parse_args(args)
    except SystemExit:
        console.print("[red]Usage:[/red] /setup [--tier T1|T2|T3] [--captioning]")
        return

    tier = parsed.tier or config.resolved_tier

    # Locate setup_models.py (project root, relative to this file's package)
    pkg_root = Path(__file__).parent.parent.parent  # rag/commands/ → project root
    setup_script = pkg_root / "setup_models.py"

    if not setup_script.exists():
        console.print(
            f"[red]setup_models.py not found[/red] at {setup_script}.\n"
            "If installed via 'motif', run: [bold]motif setup[/bold] instead."
        )
        return

    cmd = [sys.executable, str(setup_script), "--tier", tier]
    if parsed.captioning:
        cmd.append("--captioning")

    console.print(f"[dim]Running model download for Tier {tier}…[/dim]\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Setup failed[/red] (exit code {exc.returncode}).")
    except KeyboardInterrupt:
        console.print("\n[dim]Setup interrupted.[/dim]")
