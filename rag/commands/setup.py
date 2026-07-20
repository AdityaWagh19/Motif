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
        console.print("[error]Usage:[/error] /setup [--tier T1|T2|T3] [--captioning]")
        return

    tier = parsed.tier or config.resolved_tier

    # Try to import setup_models. If not available, temporarily add pkg_root to sys.path
    pkg_root = Path(__file__).parent.parent.parent
    try:
        import setup_models
    except ImportError:
        sys.path.insert(0, str(pkg_root))
        try:
            import setup_models
        except ImportError:
            console.print(
                "[error]setup_models.py not found[/error].\n"
                "Ensure you are running from the Motif root."
            )
            sys.path.pop(0)
            return
        sys.path.pop(0)

    old_argv = sys.argv.copy()
    sys.argv = ["setup_models.py", "--tier", tier]
    if parsed.captioning:
        sys.argv.append("--captioning")

    console.print(f"[structure]Running model download for Tier {tier}…[/structure]\n")
    try:
        setup_models.main()
    except Exception as exc:
        console.print(f"[error]Setup failed:[/error] {exc}")
    except KeyboardInterrupt:
        console.print("\n[structure]Setup interrupted.[/structure]")
    finally:
        sys.argv = old_argv
